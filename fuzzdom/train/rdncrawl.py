import torch
import os
import gym
import copy
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from a2c_ppo_acktr import algo, utils
from fuzzdom.models import GraphPolicy, Instructor, autoencoder_x
from fuzzdom.env import CrawlTaskEnvironment, open_driver
from fuzzdom.storage import RandomizedReplayStorage
from fuzzdom.rdn import make_rdn_vec_envs, RDNScorer, NormalizeScore
from fuzzdom.factory_resolver import FactoryResolver
from torch_geometric.data import Data, Batch
from torch_geometric.utils import train_test_split_edges, remove_self_loops

from .graph import *


class ModelBasedLeafFilter:
    def __init__(self, actor_critic, k=20):
        self.actor_critic = actor_critic.to("cpu")
        self.k = k

    def _wrap_data(self, d):
        return Batch.from_data_list([d])

    def __call__(self, leaves, dom, field, dom_field, **kwargs):
        if len(leaves) <= self.k:
            return leaves
        fr = FactoryResolver(
            self.actor_critic.base,
            dom=self._wrap_data(dom),
            _field=self._wrap_data(field),
            dom_field=self._wrap_data(dom_field),
        )
        with torch.no_grad():
            mask = fr("leaf_selector").flatten()
            # create anti-leaves mask
            d = torch.ones(mask.shape[0], dtype=torch.long)
            d[torch.tensor(leaves)] = 0
            # non-leaves set to 0
            mask[d] = 0.0
            assert mask.max().item() > 0, "Leaf selector returned no active leaves"
            topk = torch.topk(mask, self.k)
            return topk.indices.tolist()


def autoencoder_score_norm():
    return NormalizeScore(shift_mean=True, scale_by=0.5, clamp_by=(0, 0.5), alpha=0.5)


def rdn_scorer(device, text_embed_size, autoencoder_size, encoder_size):
    return (
        RDNScorer(
            autoencoder_size,
            encoder_size,
            shift_mean=True,
            scale_by=0.5,
            clamp_by=(0, 0.5),
            alpha=0.5,
        )
        .to(device)
        .eval()
    )


def filter_leaves(actor_critic):
    return ModelBasedLeafFilter(copy.deepcopy(actor_critic))


def envs(
    args, receipts, rdn_scorer, autoencoder, autoencoder_score_norm, filter_leaves
):
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
            wait_ms=500, start_url=start_url, valid_url=valid_url
        )
        return env

    torch.set_num_threads(1)

    envs = make_rdn_vec_envs(
        [make_env() for i in range(args.num_processes)],
        receipts=receipts,
        rdn_scorer=rdn_scorer,
        autoencoder=autoencoder,
        autoencoder_score_norm=autoencoder_score_norm,
        filter_leaves=filter_leaves,
    )
    return envs


def actor_critic_base():
    return Instructor


def rdn_optimizer(rdn_scorer, args):
    return torch.optim.Adam(rdn_scorer.parameters(), lr=args.rdn_lr)


def autoencoder_optimizer(autoencoder, args):
    return torch.optim.Adam(autoencoder.parameters(), lr=args.lr)


def autoencoder_storage(device, rdn_scorer):
    identifier = lambda dom: hash(rdn_scorer.actual_dom(dom, autoencoder_x(dom)))
    return RandomizedReplayStorage(
        lambda state: identifier(state[0].to(device)), device=device, alpha=0.33
    )


def rdn_storage(device, rdn_scorer):
    identifier = lambda dom, logs: hash(
        rdn_scorer.actual_dom(dom, autoencoder_x(dom))
    ) + hash(rdn_scorer.actual_logs(logs))
    return RandomizedReplayStorage(
        lambda state: identifier(state[0].to(device), state[1].to(device)),
        # device=device,
    )


def optimize(
    args,
    actor_critic,
    agent,
    rollouts,
    autoencoder,
    rdn_scorer,
    receipts,
    rdn_optimizer,
    autoencoder_optimizer,
    autoencoder_score_norm,
    filter_leaves,
    device,
    resolver,
    autoencoder_storage,
    rdn_storage,
):
    print("optimizing rdn")
    with torch.no_grad():
        next_value = actor_critic.get_value(
            rollouts.obs[-1], rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]
        ).detach()

    rollouts.compute_returns(
        next_value,
        args.use_gae,
        args.gamma,
        args.gae_lambda,
        args.use_proper_time_limits,
    )
    value_loss, action_loss, dist_entropy = agent.update(rollouts)

    ae_good_values = list(
        map(
            lambda dom: (dom,),
            filter(
                lambda dom: remove_self_loops(dom.dom_edge_index)[0].shape[1] > 4,
                map(lambda state: state[0], receipts._data.values()),
            ),
        )
    )
    autoencoder_storage.insert(ae_good_values)
    del ae_good_values

    rdn_values = list(
        map(lambda state: (state[0], state[-1]), receipts._data.values()),
    )
    rdn_storage.insert(rdn_values)
    del rdn_values

    mini_batch_size = args.num_mini_batch
    rdn_sample = list(rdn_storage.next())
    rdn_dom = Batch.from_data_list(list(map(lambda x: x[0], rdn_sample))).to(device)
    rdn_logs = Batch.from_data_list(list(map(lambda x: x[1], rdn_sample))).to(device)

    rdn_scorer.train()
    rdn_optimizer.zero_grad()
    autoencoder.train()
    autoencoder_optimizer.zero_grad()
    autoencoder_loss = None
    rdn_loss = None
    ae_values = list()

    # train rdn_scorer
    # print(rdn_dom)
    # print(rdn_logs)
    rdn_loss, rdn_scores = rdn_scorer.loss(rdn_dom, rdn_logs)
    rdn_loss.backward()

    # train autoencoder
    for data in autoencoder_storage.next():
        # no validation ?
        # data = train_test_split_edges(data)
        data = data[0]
        x = autoencoder_x(data)
        z = autoencoder.encode(x, data.dom_edge_index, data.pos_edge_index)
        autoencoder_loss = autoencoder.recon_loss(
            z, data.dom_edge_index, data.pos_edge_index
        )
        autoencoder_loss.backward()
        ae_values.append(autoencoder_loss.clone().detach().view(1))
    autoencoder_optimizer.step()
    autoencoder.eval()

    rdn_optimizer.step()
    rdn_scorer.eval()
    rdn_scorer.score_normalizer.update(rdn_scores)
    if len(ae_values) >= mini_batch_size:
        autoencoder_score_norm.update(torch.cat(ae_values))

    filter_leaves.actor_critic.load_state_dict(actor_critic.state_dict())

    resolver.update(
        {
            "value_loss": value_loss,
            "action_loss": action_loss,
            "dist_entropy": dist_entropy,
            "autoencoder_loss": autoencoder_loss,
            "rdn_loss": rdn_loss,
        }
    )


if __name__ == "__main__":
    train(locals())
