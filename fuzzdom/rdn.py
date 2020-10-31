import gym
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from .models import Encoder, ResolveMixin
from .vec_env import make_vec_envs


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    for m in model.children():
        freeze_model(m)
    return model


class RDNScorer(ResolveMixin, torch.nn.Module):
    def __init__(self, in_channels=50, out_channels=32, text_embed_size=25):
        super().__init__()
        self.dom_guesser = Encoder("GAE", in_channels, out_channels)
        self.dom_target = freeze_model(Encoder("GAE", in_channels, out_channels))
        self.log_guesser = nn.Sequential(
            nn.Linear(text_embed_size * 2, text_embed_size),
            nn.ReLU(),
            nn.Linear(text_embed_size, out_channels),
        )
        self.log_target = freeze_model(
            nn.Sequential(
                nn.Linear(text_embed_size * 2, text_embed_size),
                nn.ReLU(),
                nn.Linear(text_embed_size, out_channels),
            )
        )
        self.std_dev = torch.tensor(100.0)
        self.mean = torch.tensor(0.0)

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

    def forward(self, dom, logs):
        self.start_resolve({"dom": dom, "logs": logs})
        return self.resolve_value(self.raw_score)

    def guess_dom(self, dom, x):
        return global_mean_pool(self.dom_guesser(x, dom.dom_edge_index), dom.batch)

    def actual_dom(self, dom, x):
        with torch.no_grad():
            return global_mean_pool(self.dom_target(x, dom.dom_edge_index), dom.batch)

    def guess_logs(self, logs):
        return global_mean_pool(self.log_guesser(logs.x), logs.batch)

    def actual_logs(self, logs):
        with torch.no_grad():
            return global_mean_pool(self.log_target(logs.x), logs.batch)

    def raw_score(self, guess_logs, actual_logs, guess_dom, actual_dom):
        return F.pairwise_distance(actual_dom, guess_dom) + F.pairwise_distance(
            actual_logs, guess_logs
        )

    def score(self, dom, logs):
        self.start_resolve({"dom": dom, "logs": logs})
        scores = self.resolve_value(self.raw_score)
        return scores / self.std_dev.clamp(0.0001, 1000)

    def loss(self, dom, logs):
        self.start_resolve({"dom": dom, "logs": logs})
        scores = self.resolve_value(self.raw_score)
        loss = scores.mean()
        with torch.no_grad():
            # TODO momentum?
            self.std_dev = (scores.std() + self.std_dev) / 2
            self.mean = (loss + self.mean) / 2
        return loss


class RDNScorerGymWrapper(gym.Wrapper):
    def __init__(self, env, rdn_scorer):
        super().__init__(env)
        self.rdn_scorer = rdn_scorer

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        dom = observation[0]
        logs = observation[-1]
        with torch.no_grad():
            score = self.rdn_scorer.score(dom, logs)
        reward = score.item()
        return observation, reward, done, info


def make_rdn_vec_envs(envs, receipts, rdn_scorer, filter_leaves=None):

    return make_vec_envs(
        envs,
        receipts,
        inner=lambda env: RDNScorerGymWrapper(env, rdn_scorer),
        filter_leaves=filter_leaves,
    )
