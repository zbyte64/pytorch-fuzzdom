import os, sys
import copy
import glob
import time
from collections import deque

import gym
import numpy as np
import torch

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import Policy

from fuzzdom.env import MiniWoBGraphEnvironment
from fuzzdom.models import GNNBase
from fuzzdom.storage import StorageReceipt, ReceiptRolloutStorage
from fuzzdom.vec_env import (
    GraphGymWrapper,
    ReceiptsGymWrapper,
    make_vec_envs,
    GraphActionWrapper,
)
from fuzzdom.distributions import NodeObjective
from fuzzdom.curriculum import LevelTracker, MINIWOB_CHALLENGES
from fuzzdom.dir_paths import MINIWOB_HTML


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    receipts = StorageReceipt()
    make_env = lambda tasks: MiniWoBGraphEnvironment(
        base_url=os.environ.get("BASE_URL", f"file://{MINIWOB_HTML}/"),
        levels=tasks,
        level_tracker=LevelTracker(tasks),
        wait_ms=500,
    )

    task = args.env_name
    if args.env_name == "PongNoFrameskip-v4":
        args.env_name = "clickbutton"
        task = "miniwob/click-button.html"
    if task == "levels":
        tasks = MINIWOB_CHALLENGES
    else:
        tasks = [[task]]
    print("Selected tasks:", tasks)
    NUM_ACTIONS = 1
    envs = make_vec_envs([make_env(tasks[i % len(tasks)]) for i in range(args.num_processes)], receipts)

    if os.path.exists("./datadir/autoencoder.pt"):
        dom_autoencoder = torch.load("./datadir/autoencoder.pt")
        dom_encoder = dom_autoencoder.encoder
        for param in dom_encoder.parameters():
            param.requires_grad = False
    else:
        print("No dom encoder")
        dom_encoder = None
    actor_critic = Policy(
        envs.observation_space.shape,
        gym.spaces.Discrete(NUM_ACTIONS),  # envs.action_space,
        base=GNNBase,
        base_kwargs={"dom_encoder": dom_encoder, "recurrent": args.recurrent_policy},
    )
    actor_critic.dist = NodeObjective()
    actor_critic.to(device)

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

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100, device
        )
        file_name = os.path.join(
            args.gail_experts_dir,
            "trajs_{}.pt".format(args.env_name.split("-")[0].lower()),
        )

        gail_train_loader = torch.utils.data.DataLoader(
            rr.get_dataset(),
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=True,
        )

    from tensorboardX import SummaryWriter
    import datetime

    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    tensorboard_writer = SummaryWriter(log_dir=os.path.join("/tmp/log", ts_str))

    rollouts = ReceiptRolloutStorage(
        args.num_steps,
        args.num_processes,
        (1,),  # envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size,
        receipts,
    )

    # resume from last save
    if args.save_dir != "":
        save_path = os.path.join(args.save_dir, args.algo)
        try:
            os.makedirs(save_path)
        except OSError:
            pass

        model_path = os.path.join(save_path, args.env_name + ".pt")
        if os.path.exists(model_path):
            print("Loadng previous model:", model_path)
            actor_critic = torch.load(model_path)
            actor_critic.train()

    obs = envs.reset()
    rollouts.obs[0].copy_(torch.tensor(obs))
    rollouts.to(device)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    print("Iterations:", num_updates, args.num_steps)
    for j in range(num_updates):
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
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    receipts.redeem(rollouts.obs[step]),
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                )

            # Obser reward and next obs
            last_action_time = time.time()
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

        with torch.no_grad():
            next_value = actor_critic.get_value(
                receipts.redeem(rollouts.obs[-1]),
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                obsfilt = lambda x, update: x  # utils.get_vec_normalize(envs)._obfilt
                print(gail_train_loader)
                discr.update(gail_train_loader, rollouts, obsfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
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
        obs_shape = rollouts.obs.size()[2:]
        obs = rollouts.obs[:-1].view(-1, *obs_shape)
        obs = obs[torch.randint(0, obs.size(0), (1, 32))]

        rollouts.after_update()

        receipts.prune(rollouts.obs)

        # save for every interval-th episode or for the last epoch
        if (
            j % args.save_interval == 0 or j == num_updates - 1
        ) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            model_path = os.path.join(save_path, args.env_name + ".pt")
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
                    dist_entropy,
                    value_loss,
                    action_loss,
                )
            )

            from pprint import pprint
            
            pprint(LevelTracker.global_scoreboard)

            # tensorboard_writer.add_histogram(
            #    "task_ranks", torch.tensor(predictor._difficulty_rank), total_num_steps
            # )
            tensorboard_writer.add_histogram("value", value, total_num_steps)
            tensorboard_writer.add_histogram(
                "x", actor_critic.base.last_x, total_num_steps
            )
            tensorboard_writer.add_histogram(
                "query", actor_critic.base.last_query, total_num_steps
            )
            tensorboard_writer.add_histogram(
                "inputs_at", actor_critic.base.last_inputs_at, total_num_steps
            )

            tensorboard_writer.add_scalar(
                "mean_reward", np.mean(episode_rewards), total_num_steps
            )
            tensorboard_writer.add_scalar(
                "median_reward", np.median(episode_rewards), total_num_steps
            )
            tensorboard_writer.add_scalar(
                "min_reward", np.min(episode_rewards), total_num_steps
            )
            tensorboard_writer.add_scalar(
                "max_reward", np.max(episode_rewards), total_num_steps
            )
            tensorboard_writer.add_scalar("dist_entropy", dist_entropy, total_num_steps)
            tensorboard_writer.add_scalar("value_loss", value_loss, total_num_steps)
            tensorboard_writer.add_scalar("action_loss", action_loss, total_num_steps)

        if (
            args.eval_interval is not None
            and len(episode_rewards) > 1
            and j % args.eval_interval == 0
        ):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(
                actor_critic,
                ob_rms,
                args.env_name,
                args.seed,
                args.num_processes,
                eval_log_dir,
                device,
            )


if __name__ == "__main__":
    from pprint import pprint

    try:
        main()
    finally:
        pprint(LevelTracker.global_scoreboard)
