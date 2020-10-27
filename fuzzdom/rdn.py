import gym
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from .models import Encoder, ResolveMixin
from .vec_env import make_vec_envs


class RDNScorer(ResolveMixin, torch.nn.Module):
    def __init__(self, in_channels=50, out_channels=32):
        super().__init__()
        self.rdn_guesser = Encoder("GAE", in_channels, out_channels)
        self.rdn_target = Encoder("GAE", in_channels, out_channels).eval()  # freeze?

    def x(self, dom):
        return torch.cat(
            [
                dom.text,
                dom.value,
                dom.radio_value,
                dom.tag,
                dom.classes,
                dom.rx,
                dom.ry,
                dom.width,
                dom.height,
                dom.top,
                dom.left,
                dom.focused,
                dom.depth,
            ],
            dim=1,
        )

    def forward(self, dom):
        self.start_resolve({"dom": dom})
        return self.resolve_value(self.score)

    def guess(self, dom, x):
        return global_mean_pool(self.rdn_guesser(x, dom.dom_edge_index), dom.batch)

    def actual(self, dom, x):
        with torch.no_grad():
            return global_mean_pool(self.rdn_target(x, dom.dom_edge_index), dom.batch)

    def score(self, guess, actual):
        return torch.mean(F.pairwise_distance(actual, guess))


class RDNScorerGymWrapper(gym.Wrapper):
    def __init__(self, env, rdn_scorer):
        super().__init__(env)
        self.rdn_scorer = rdn_scorer

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        dom = observation[0]
        with torch.no_grad():
            score = self.rdn_scorer(dom)
        reward = score
        return observation, reward, done, info


def make_rdn_vec_envs(envs, receipts, rdn_scorer, filter_leaves=None):

    return make_vec_envs(
        envs,
        receipts,
        inner=lambda env: RDNScorerGymWrapper(env, rdn_scorer),
        filter_leaves=filter_leaves,
    )
