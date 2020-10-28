import torch
import os
import gym
import copy
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from a2c_ppo_acktr import algo, utils
from fuzzdom.models import GraphPolicy, Instructor
from fuzzdom.env import CrawlTaskEnvironment, open_driver
from .graph import RunTime as BaseRunTime
from fuzzdom.rdn import make_rdn_vec_envs, RDNScorer
from fuzzdom.factory_resolver import FactoryResolver
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges, remove_self_loops


class ModelBasedLeafFilter:
    def __init__(self, actor_critic, k=20):
        self.actor_critic = actor_critic.to("cpu")
        self.k = k

    def __call__(self, leaves, dom, field, dom_field, **kwargs):
        if len(leaves) <= self.k:
            return leaves
        fr = FactoryResolver(
            self.actor_critic.base, dom=dom, _field=field, dom_field=dom_field
        )
        with torch.no_grad():
            # TODO: use dom interest?
            disabled_mask = 1 - fr(self.actor_critic.base.disabled_mask).flatten()
            d = torch.ones(disabled_mask.shape[0], dtype=torch.long)
            d[torch.tensor(leaves)] = 0
            d = 1 - d
            disabled_mask[d] = 0.0
            topk = torch.topk(disabled_mask, self.k)
            return topk.indices.tolist()


class RunTime(BaseRunTime):
    def rdn_scorer(self, device):
        in_size = self.text_embed_size * 4 + 9
        out_size = self.encoder_size
        return RDNScorer(in_size, out_size).to(device)

    def filter_leaves(self, actor_critic):
        return ModelBasedLeafFilter(copy.deepcopy(actor_critic))

    def envs(self, args, receipts, rdn_scorer, filter_leaves):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        log_dir = os.path.expanduser(args.log_dir)
        eval_log_dir = log_dir + "_eval"
        utils.cleanup_log_dir(log_dir)
        utils.cleanup_log_dir(eval_log_dir)

        start_url = args.env_name
        valid_url = lambda x: True  # x.startswith(start_url)

        def make_env():
            env = CrawlTaskEnvironment(
                wait_ms=500, start_url=start_url, valid_url=valid_url,
            )
            return env

        torch.set_num_threads(1)

        envs = make_rdn_vec_envs(
            [make_env() for i in range(args.num_processes)],
            receipts,
            rdn_scorer,
            filter_leaves=filter_leaves,
        )
        return envs

    def actor_critic(self, args, device, receipts, dom_encoder):
        if args.load_model:
            print("Loadng previous model:", args.load_model)
            actor_critic = torch.load(args.load_model)
            actor_critic.receipts = receipts
        else:
            actor_critic = GraphPolicy(
                (args.num_processes,),
                gym.spaces.Discrete(1),
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
        filter_leaves,
        device,
    ):
        print("optimizing")
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

        # TODO lookup batch size
        mini_batch_size = 16
        sampler = BatchSampler(
            SubsetRandomSampler(rollouts.obs.flatten().tolist()),
            mini_batch_size,
            drop_last=True,
        )
        autoencoder_subset = None
        rdn_scorer.train()
        rdn_optimizer.zero_grad()
        for subset in sampler:
            if autoencoder_subset is None:
                autoencoder_subset = subset
            doms = receipts.redeem(torch.tensor(subset))[0]
            doms.edge_index = doms.dom_edge_index
            # train rdn_scorer
            rdn_loss = rdn_scorer(doms)
            rdn_loss.backward()
        rdn_optimizer.step()

        autoencoder_loss = None
        if autoencoder_subset is not None:
            # train autoencoder
            autoencoder.train()
            autoencoder_optimizer.zero_grad()
            for i in autoencoder_subset:
                data = receipts[i][0]
                del data.batch
                data.edge_index = data.dom_edge_index
                # skip empty/small edges
                check, _ = remove_self_loops(data.edge_index)
                if check.shape[1] < 5:
                    continue
                # no validation ?
                # data = train_test_split_edges(data)
                x = rdn_scorer.x(data)
                z = autoencoder.encode(x, data.edge_index)
                autoencoder_loss = autoencoder.recon_loss(z, data.edge_index)
                autoencoder_loss.backward()
            autoencoder_optimizer.step()

        filter_leaves.actor_critic.load_state_dict(actor_critic.state_dict())

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
