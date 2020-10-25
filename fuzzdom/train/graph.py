import os, sys
import copy
import glob
import time
from collections import deque
import datetime

import gym
import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import GAE

from tensorboardX import SummaryWriter
from a2c_ppo_acktr import algo, utils
from fuzzdom.arguments import get_args
from a2c_ppo_acktr.storage import RolloutStorage

from fuzzdom.factory_resolver import FactoryResolver
from fuzzdom.env import MiniWoBGraphEnvironment, ManagedWebInterface
from fuzzdom.models import GNNBase, GraphPolicy, Encoder
from fuzzdom.storage import StorageReceipt
from fuzzdom.vec_env import make_vec_envs
from fuzzdom.curriculum import LevelTracker, MINIWOB_CHALLENGES
from fuzzdom.dir_paths import MINIWOB_HTML
from fuzzdom import gail
from fuzzdom.replay import ReplayRepository


class RunTime:
    def __init__(self):
        self.text_embed_size = 25
        self.encoder_size = 25

    def args(self):
        return get_args()

    def envs(self, args, receipts):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        log_dir = os.path.expanduser(args.log_dir)
        eval_log_dir = log_dir + "_eval"
        utils.cleanup_log_dir(log_dir)
        utils.cleanup_log_dir(eval_log_dir)

        torch.set_num_threads(1)
        make_env = lambda tasks: MiniWoBGraphEnvironment(
            base_url=os.environ.get("BASE_URL", f"file://{MINIWOB_HTML}/"),
            levels=tasks,
            level_tracker=LevelTracker(tasks),
            wait_ms=500,
            web_interface=ManagedWebInterface(proxy=os.getenv("PROXY_HOST")),
        )

        task = args.env_name
        if task == "levels":
            tasks = MINIWOB_CHALLENGES
        else:
            tasks = [[task]]
        print("Selected tasks:", tasks)
        envs = make_vec_envs(
            [make_env(tasks[i % len(tasks)]) for i in range(args.num_processes)],
            receipts,
        )
        return envs

    def receipts(self, device):
        return StorageReceipt(device)

    def device(self, args):
        if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        return torch.device("cuda:0" if args.cuda else "cpu")

    def autoencoder(self, args):
        if os.path.exists("./datadir/autoencoder.pt"):
            return torch.load("./datadir/autoencoder.pt")
        in_size = self.text_embed_size * 4 + 9
        out_size = self.encoder_size
        return GAE(Encoder("GAE", in_size, out_size))

    def dom_encoder(self, autoencoder):
        return autoencoder.encoder if autoencoder else None

    def actor_critic(self, envs, args, device, receipts, dom_encoder):
        if args.load_model:
            print("Loadng previous model:", args.load_model)
            actor_critic = torch.load(args.load_model)
            actor_critic.receipts = receipts
        else:
            if dom_encoder:
                for param in dom_encoder.parameters():
                    param.requires_grad = False
            else:
                print("No dom encoder")
                dom_encoder = None
            actor_critic = GraphPolicy(
                envs.observation_space.shape,
                gym.spaces.Discrete(1),  # envs.action_space,
                base=GNNBase,
                base_kwargs={
                    "dom_encoder": dom_encoder,
                    "recurrent": args.recurrent_policy,
                },
                receipts=receipts,
            )
        actor_critic.to(device)
        actor_critic.train()
        if actor_critic.base.dom_encoder:
            actor_critic.base.dom_encoder.eval()
        return actor_critic

    def agent(self, args, actor_critic):
        if args.algo == "a2c":
            agent = algo.A2C_ACKTR(
                actor_critic,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                max_grad_norm=args.max_grad_norm,
            )
        elif args.algo == "ppo":
            agent = algo.PPO(
                actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm,
            )
        elif args.algo == "acktr":
            agent = algo.A2C_ACKTR(
                actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True
            )
        return agent

    def tensorboard_writer(self):
        ts_str = datetime.datetime.fromtimestamp(time.time()).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        return SummaryWriter(log_dir=os.path.join("/tmp/log", ts_str))

    def rollouts(self, args, envs, actor_critic):
        return RolloutStorage(
            args.num_steps,
            args.num_processes,
            (1,),  # envs.observation_space.shape,
            envs.action_space,
            actor_critic.recurrent_hidden_state_size,
        )

    def model_path(self, args):
        if args.save_dir != "" and args.save_interval:
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            return os.path.join(save_path, args.env_name + ".pt")

    def num_updates(self, args):
        return int(args.num_env_steps) // args.num_steps // args.num_processes

    def __call__(self):
        resolv = FactoryResolver(self)
        print("Initializing...")
        resolv.update(resolv(self.start))
        num_updates = resolv["num_updates"]
        args = resolv["args"]
        print("Iterations:", num_updates, args.num_steps)
        last_action_time = time.time()

        for j in range(num_updates):
            resolv["j"] = j
            resolv["last_action_time"] = last_action_time
            resolv.update(resolv(self.run_episode))
            last_action_time = time.time()
            resolv.update(resolv(self.optimize))
            resolv(self.episode_tick)

    def start(self, envs, rollouts, device):
        obs = envs.reset()
        rollouts.obs[0].copy_(torch.tensor(obs))
        rollouts.to(device)
        start = time.time()
        return {"start": start, "obs": obs}

    def run_episode(
        self,
        args,
        envs,
        rollouts,
        actor_critic,
        agent,
        j,
        num_updates,
        last_action_time,
        obs,
    ):
        episode_rewards = deque(maxlen=args.num_steps * args.num_processes)
        if j and last_action_time + 5 < time.time():
            # task likely timed out
            print("Reseting tasks")
            obs = envs.reset()
            rollouts.obs[0].copy_(torch.tensor(obs))
            rollouts.recurrent_hidden_states[0].copy_(
                torch.zeros_like(rollouts.recurrent_hidden_states[0])
            )
            rollouts.masks[0].copy_(torch.zeros_like(rollouts.masks[0]))

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer,
                j,
                num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr,
            )

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                (
                    value,
                    action,
                    action_log_prob,
                    recurrent_hidden_states,
                ) = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                )

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for e, i in enumerate(infos):
                if i.get("real_action") is not None:
                    action[e] = i["real_action"]
                if i.get("bad_transition"):
                    action[e] = torch.zeros_like(action[e])

            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )
            rollouts.insert(
                torch.tensor(obs),
                recurrent_hidden_states,
                action,
                action_log_prob,
                value,
                torch.tensor(reward).unsqueeze(1),
                masks,
                bad_masks,
            )

        return {"obs": obs, "episode_rewards": episode_rewards}

    def optimize(self, args, actor_critic, agent, rollouts):
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(
            next_value,
            args.use_gae,
            args.gamma,
            args.gae_lambda,
            args.use_proper_time_limits,
        )
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        return {
            "value_loss": value_loss,
            "action_loss": action_loss,
            "dist_entropy": dist_entropy,
        }

    def episode_tick(
        self,
        args,
        j,
        num_updates,
        start,
        episode_rewards,
        actor_critic,
        tensorboard_writer,
        receipts,
        rollouts,
        resolver,
    ):
        # put last observation first and clear
        obs_shape = rollouts.obs.size()[2:]
        obs = rollouts.obs[:-1].view(-1, *obs_shape)
        obs = obs[torch.randint(0, obs.size(0), (1, 32))]
        resolver["obs"] = obs

        rollouts.after_update()

        receipts.prune(rollouts.obs)

        # save for every interval-th episode or for the last epoch
        if (
            args.save_interval
            and args.save_dir != ""
            and (j % args.save_interval == 0 or j == num_updates - 1)
        ):
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            model_path = os.path.join(save_path, "agent.pt")
            torch.save(actor_critic, model_path)
            print("Saved model:", model_path)

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                    # dist_entropy,
                    # value_loss,
                    # action_loss,
                )
            )

            from pprint import pprint

            pprint(LevelTracker.global_scoreboard)

            actor_critic.base.report_values(tensorboard_writer, total_num_steps)
            resolver.report_values(tensorboard_writer, total_num_steps)


if __name__ == "__main__":
    from pprint import pprint

    runtime = RunTime()

    try:
        runtime()
    finally:
        pprint(LevelTracker.global_scoreboard)
