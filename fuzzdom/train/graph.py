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

from tensorboardX import SummaryWriter
from a2c_ppo_acktr import algo, utils
from fuzzdom.arguments import get_args
from a2c_ppo_acktr.storage import RolloutStorage

from fuzzdom.factory_resolver import FactoryResolver
from fuzzdom.env import MiniWoBGraphEnvironment, ManagedWebInterface
from fuzzdom.models import GNNBase, GraphPolicy, GAE
from fuzzdom.domx import text_embed_size
from fuzzdom.storage import StorageReceipt, TaskQueueReplayStorage
from fuzzdom.vec_env import make_vec_envs
from fuzzdom.curriculum import LevelTracker, MINIWOB_CHALLENGES
from fuzzdom.dir_paths import MINIWOB_HTML
from fuzzdom import gail
from fuzzdom.replay import ReplayRepository


encoder_size = 120
autoencoder_size = text_embed_size * 4


def args():
    return get_args()


def log_dir(args):
    return os.path.expanduser(args.log_dir)


def envs(args, receipts, log_dir):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    make_env = lambda tasks: MiniWoBGraphEnvironment(
        base_url=os.environ.get("BASE_URL", f"file://{MINIWOB_HTML}/"),
        levels=tasks,
        level_tracker=LevelTracker(tasks),
        wait_ms=500,
        web_interface=ManagedWebInterface(),
    )

    task = args.env_name
    if task == "levels":
        tasks = MINIWOB_CHALLENGES
    else:
        tasks = [[task]]
    print("Selected tasks:", tasks)
    envs = make_vec_envs(
        [make_env(tasks[i % len(tasks)]) for i in range(args.num_processes)], receipts
    )
    return envs


def receipts(device):
    return StorageReceipt(device)


def device(args):
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return torch.device("cuda:0" if args.cuda else "cpu")


def autoencoder(args, text_embed_size, encoder_size, autoencoder_size):
    if isinstance(args.load_autoencoder, str):
        return torch.load(args.load_autoencoder)
    return GAE(autoencoder_size, encoder_size)


def dom_encoder(autoencoder):
    return autoencoder


def actor_critic_base():
    return GNNBase


def actor_critic(args, device, receipts, dom_encoder, actor_critic_base):
    actor_critic = GraphPolicy(
        (args.num_processes,),
        gym.spaces.Discrete(1),  # envs.action_space,
        base=actor_critic_base,
        base_kwargs={"dom_encoder": dom_encoder, "recurrent": args.recurrent_policy},
        receipts=receipts,
    )
    if args.load_model:
        print("Loading previous model:", args.load_model)
        loaded_model = torch.load(args.load_model)
        # chomp state down from rdncrawl
        new_state = loaded_model.state_dict()
        cur_state = actor_critic.state_dict()

        # load_autoencoder
        all_keys = set(filter(lambda k: k in new_state, cur_state.keys()))
        ae_keys = set(filter(lambda key: key.startswith("base.dom_encoder"), all_keys))
        # load_actor
        # load_critic
        critic_keys = set(filter(lambda key: key.startswith("base.critic"), all_keys))
        actor_keys = all_keys - ae_keys - critic_keys
        if args.load_actor:
            assert actor_keys
            for key in actor_keys:
                cur_state[key] = new_state[key]

        if args.load_autoencoder is True:
            assert ae_keys, str(all_keys)
            for key in ae_keys:
                cur_state[key] = new_state[key]

        if args.load_critic:
            assert critic_keys
            for key in critic_keys:
                cur_state[key] = new_state[key]

        actor_critic.load_state_dict(cur_state)

    actor_critic.to(device)
    actor_critic.train()
    return actor_critic


def agent(args, actor_critic):
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


def tensorboard_writer(log_dir):
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    return SummaryWriter(log_dir=os.path.join(log_dir, "log", ts_str))


def rollouts(args, envs, actor_critic):
    return RolloutStorage(
        args.num_steps,
        args.num_processes,
        (1,),  # envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size,
    )


def num_updates(args):
    return int(args.num_env_steps) // args.num_steps // args.num_processes


def start(envs, rollouts, device, resolver):
    obs = envs.reset()
    rollouts.obs[0].copy_(torch.tensor(obs))
    rollouts.to(device)
    start = time.time()
    resolver.update({"start": start, "obs": obs})


def run_episode(
    args,
    envs,
    rollouts,
    actor_critic,
    agent,
    j,
    num_updates,
    last_action_time,
    obs,
    gail_train_loader,
    receipts,
    resolver,
):
    episode_rewards = deque(maxlen=args.num_steps * args.num_processes)
    rewards = deque(maxlen=args.num_steps * args.num_processes)
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
    start_episode_step = [0 for i in range(args.num_processes)]

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

        rewards.append(reward)

        for e, info in enumerate(infos):
            if info.get("bad_transition"):
                action[e] = torch.zeros_like(action[e])
            elif "episode" in info.keys():
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

        if args.gail:
            for e, info in enumerate(infos):
                if "episode" in info.keys() and info["episode"]["r"] > 0:
                    series = [
                        receipts._data[rollouts.obs[i, e].item()]
                        for i in range(start_episode_step[e], step + 1)
                    ]
                    actions = rollouts.actions[
                        start_episode_step[e] : step + 1, e, 0
                    ].tolist()
                    gail_train_loader.store_episode(info["task"], series, actions)
                    start_episode_step[e] = step

    resolver.update(
        {
            "obs": obs,
            "episode_rewards": episode_rewards,
            "rewards": torch.tensor(rewards),
        }
    )


def gail_discr(args, envs, actor_critic_base, dom_encoder, device):
    if args.gail:
        return gail.Discriminator(
            actor_critic_base(input_dim=None, dom_encoder=dom_encoder, recurrent=False),
            device,
        )


def gail_train_loader(args, device):
    if args.gail:
        return TaskQueueReplayStorage(args.num_mini_batch, 10 * args.num_steps, device)


def optimize(
    args,
    actor_critic,
    agent,
    rollouts,
    j,
    gail_train_loader,
    gail_discr,
    receipts,
    resolver,
):
    with torch.no_grad():
        next_value = actor_critic.get_value(
            rollouts.obs[-1], rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]
        ).detach()

    if args.gail and len(gail_train_loader) >= gail_train_loader.batch_size:
        gail_epoch = args.gail_epoch
        if j < 10:
            gail_epoch = 100  # Warm up
        for _ in range(gail_epoch):
            gail_discr.update(gail_train_loader, rollouts, receipts)

        for step in range(args.num_steps):
            rollouts.rewards[step] = gail_discr.predict_reward(
                receipts.redeem(rollouts.obs[step]),
                rollouts.actions[step],
                args.gamma,
                rollouts.masks[step],
            )

    rollouts.compute_returns(
        next_value,
        args.use_gae,
        args.gamma,
        args.gae_lambda,
        args.use_proper_time_limits,
    )
    value_loss, action_loss, dist_entropy = agent.update(rollouts)
    resolver.update(
        {
            "value_loss": value_loss,
            "action_loss": action_loss,
            "dist_entropy": dist_entropy,
        }
    )


def save_models(args, actor_critic):
    save_path = args.save_dir
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    torch.save(actor_critic, args.save_model)
    print("Saved model:", args.save_model)


def log_stats(args, j, episode_rewards, start, value_loss, action_loss):
    total_num_steps = (j + 1) * args.num_processes * args.num_steps
    if len(episode_rewards) > 1:
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
    else:
        print(
            "Updates {j}, value loss {value_loss}, action loss {action_loss}".format(
                j=j, value_loss=value_loss, action_loss=action_loss,
            )
        )

    if hasattr(LevelTracker, "global_scoreboard") and len(episode_rewards) > 1:
        from pprint import pprint

        pprint(LevelTracker.global_scoreboard)


def episode_tick(
    args, j, num_updates, receipts, rollouts, resolver,
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
        and args.save_dir
        and (j % args.save_interval == 0 or j == num_updates - 1)
    ):
        resolver("save_models")

    if args.log_interval and j % args.log_interval == 0:
        resolver("log_stats")

    # attempt to fix segfault
    # https://github.com/pytorch/pytorch/issues/19969
    if args.cuda:
        torch.cuda.synchronize()


def train(modules=locals()):
    r = FactoryResolver(modules)
    args = r["args"]
    if args.profile_memory:
        import tracemalloc

        tracemalloc.start()
    print("Initializing...")
    r("start")
    num_updates = r["num_updates"]

    print("Iterations:", num_updates, args.num_steps)
    last_action_time = time.time()

    if args.profile_memory:
        last_snapshot = tracemalloc.take_snapshot()

    tensorboard_writer = r["tensorboard_writer"]

    for j in range(num_updates):
        with r.chain(j=j, last_action_time=last_action_time) as s:
            s("run_episode")
            last_action_time = time.time()
            if args.log_interval and j % args.log_interval == 0:
                s.enable_reporting(tensorboard_writer, j)
            s("optimize")
            s.report_values()
            s.disable_reporting()
            s("episode_tick")
            r["obs"] = s["obs"]
        if args.profile_memory:
            curr_snapshot = tracemalloc.take_snapshot()
            stats = curr_snapshot.compare_to(last_snapshot, "lineno")
            # display top 5
            for i, stat in enumerate(stats[:5]):
                print(i, stat)
            last_snapshot = curr_snapshot


if __name__ == "__main__":
    from pprint import pprint

    try:
        train()
    finally:
        pprint(LevelTracker.global_scoreboard)
