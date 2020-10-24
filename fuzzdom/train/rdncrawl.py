import torch
import os
import gym
from a2c_ppo_acktr import algo, utils
from fuzzdom.models import GraphPolicy, Instructor
from fuzzdom.env import CrawlTaskEnvironment, open_driver
from .graph import RunTime as BaseRunTime
from fuzzdom.rdn import make_rdn_vec_envs, RDNScorer


class RunTime(BaseRunTime):
    def rdn_scorer(self):
        return RDNScorer()

    def envs(self, args, receipts, rdn_scorer):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        log_dir = os.path.expanduser(args.log_dir)
        eval_log_dir = log_dir + "_eval"
        utils.cleanup_log_dir(log_dir)
        utils.cleanup_log_dir(eval_log_dir)

        start_url = args.env_name
        valid_url = lambda x: x.startswith(start_url)

        def make_env():
            env = CrawlTaskEnvironment(
                wait_ms=500, start_url=start_url, valid_url=valid_url
            )
            return env

        torch.set_num_threads(1)

        envs = make_rdn_vec_envs(
            [make_env() for i in range(args.num_processes)], receipts, rdn_scorer
        )
        return envs

    def actor_critic(self, envs, args, device, receipts, dom_encoder):
        if args.load_model:
            print("Loadng previous model:", args.load_model)
            actor_critic = torch.load(args.load_model)
            actor_critic.receipts = receipts
        else:
            actor_critic = GraphPolicy(
                envs.observation_space.shape,
                gym.spaces.Discrete(1),  # envs.action_space,
                base=Instructor,
                base_kwargs={
                    "dom_encoder": dom_encoder,
                    "recurrent": args.recurrent_policy,
                },
                receipts=receipts,
            )
        actor_critic.to(device)
        actor_critic.train()
        return actor_critic

    def rdn_optimizer(self, rdn_scorer):
        return torch.optim.Adam(rdn_scorer.parameters(), lr=0.01)

    def autoencoder_optimizer(self, autoencoder):
        return torch.optim.Adam(autoencoder.parameters(), lr=0.01)

    def optimize(
        self,
        args,
        actor_critic,
        agent,
        rollouts,
        autoencoder,
        rdn_scorer,
        receipts,
        rdn_optimizer,
        autoencoder_optimizer,
    ):
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

        all_doms = map(lambda x: x[0], receipts[rollouts.obs])
        all_doms.edge_index = all_doms.dom_edge_index

        # train autoencoder
        autoencoder.train()
        autoencoder_optimizer.zero_grad()
        data = train_test_split_edges(all_doms)
        x = rdn_scorer.x(all_doms)
        z = autoencoder.encode(x, data.train_pos_edge_index)
        autoencoder_loss = autoencoder.recon_loss(z, data.train_pos_edge_index)
        autoencoder_loss.backward()
        autoencoder_optimizer.step()

        # train rdn_scorer
        rdn_scorer.train()
        rdn_optimizer.zero_grad()
        rdn_loss = rdn_scorer(all_doms)
        rdn_loss.backward()
        rdn_optimizer.step()

        return {
            "value_loss": value_loss,
            "action_loss": action_loss,
            "dist_entropy": dist_entropy,
            "autoencoder_loss": autoencoder_loss,
            "rdn_loss": rdn_loss,
        }


if __name__ == "__main__":
    runtime = RunTime()
    runtime()
