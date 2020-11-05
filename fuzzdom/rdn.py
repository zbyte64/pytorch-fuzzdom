import gym
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from .models import Encoder, ResolveMixin, autoencoder_x
from .vec_env import make_vec_envs
from .functions import init_xu


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    for m in model.children():
        freeze_model(m)
    return model


class NormalizeScore:
    def __init__(self, alpha=0.8, scale_by=1, clamp_by=(0, 1), shift_mean=True):
        self.alpha = alpha
        self.beta = 1 - alpha
        self.scale_by = scale_by
        self.clamp_by = clamp_by
        self.shift_mean = shift_mean
        self.std_dev = None
        self.mean = None

    def update(self, scores):
        with torch.no_grad():
            if self.shift_mean:
                mean = scores.mean()
            else:
                mean = 0
            std_dev = scores.std()
            if self.std_dev is None:
                self.std_dev = std_dev
                self.mean = mean
            else:
                self.std_dev = self.std_dev * self.alpha + std_dev * self.beta
                self.mean = self.mean * self.alpha + mean * self.beta

    def scale(self, score):
        if self.std_dev is None:
            if self.shift_mean:
                return torch.zeros_like(score)
            return (score * self.scale_by).clamp(*self.clamp_by)
        return (
            (score - (self.mean if self.shift_mean else 0))
            * self.scale_by
            / self.std_dev
        ).clamp(*self.clamp_by)


class RDNScorer(ResolveMixin, torch.nn.Module):
    def __init__(self, in_channels=50, out_channels=32, text_embed_size=25):
        super().__init__()
        self.dom_guesser = Encoder("GAE", in_channels, out_channels)
        self.dom_target = freeze_model(Encoder("GAE", in_channels, out_channels))
        self.log_guesser = nn.Sequential(
            init_xu(nn.Linear(text_embed_size * 2, text_embed_size), "relu"),
            nn.ReLU(),
            init_xu(nn.Linear(text_embed_size, out_channels)),
        )
        self.log_target = freeze_model(
            nn.Sequential(
                init_xu(nn.Linear(text_embed_size * 2, text_embed_size), "relu"),
                nn.ReLU(),
                init_xu(nn.Linear(text_embed_size, out_channels)),
            )
        )
        self.score_normalizer = NormalizeScore(
            shift_mean=False, scale_by=0.5, clamp_by=(0, 2)
        )

    def x(self, dom):
        return autoencoder_x(dom)

    def forward(self, dom, logs):
        self.start_resolve({"dom": dom, "logs": logs})
        return self.resolve_value(self.raw_score)

    def guess_dom(self, dom, x):
        return global_max_pool(self.dom_guesser(x, dom.dom_edge_index), dom.batch)

    def actual_dom(self, dom, x):
        with torch.no_grad():
            return global_max_pool(self.dom_target(x, dom.dom_edge_index), dom.batch)

    def guess_logs(self, logs):
        return global_max_pool(self.log_guesser(logs.x), logs.batch)

    def actual_logs(self, logs):
        with torch.no_grad():
            return global_max_pool(self.log_target(logs.x), logs.batch)

    def raw_score(self, guess_logs, actual_logs, guess_dom, actual_dom):
        return F.pairwise_distance(actual_dom, guess_dom) + F.pairwise_distance(
            actual_logs, guess_logs
        )

    def score(self, dom, logs):
        self.start_resolve({"dom": dom, "logs": logs})
        scores = self.resolve_value(self.raw_score)
        return self.score_normalizer.scale(scores)

    def loss(self, dom, logs):
        self.start_resolve({"dom": dom, "logs": logs})
        scores = self.resolve_value(self.raw_score)
        self.score_normalizer.update(scores)
        loss = scores.mean()
        return loss


class RDNScorerGymWrapper(gym.Wrapper):
    def __init__(self, env, rdn_scorer):
        super().__init__(env)
        self.rdn_scorer = rdn_scorer

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward += self.score_observation(observation)
        return observation, reward, done, info

    def score_observation(self, observation):
        dom = observation[0]
        logs = observation[-1]
        with torch.no_grad():
            score = self.rdn_scorer.score(dom, logs)
            score = score.item()
            print("RDN SCORE", score)
            if math.isnan(score):
                return 0
            return score


class AEScorerGymWrapper(gym.Wrapper):
    def __init__(self, env, autoencoder, score_normalizer):
        super().__init__(env)
        self.autoencoder = autoencoder
        self.score_normalizer = score_normalizer

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward += self.score_observation(observation)
        return observation, reward, done, info

    def score_observation(self, observation):
        dom = observation[0]
        with torch.no_grad():
            x = autoencoder_x(dom)
            z = self.autoencoder.encode(x, dom.edge_index)
            score = self.autoencoder.recon_loss(z, dom.edge_index)
            score = self.score_normalizer.scale(score)
            score = score.item()
            print("AE SCORE", score)
            if math.isnan(score):
                return 0
            return score


def make_rdn_vec_envs(
    envs, receipts, rdn_scorer, autoencoder, autoencoder_score_norm, filter_leaves
):

    return make_vec_envs(
        envs,
        receipts,
        inner=lambda env: AEScorerGymWrapper(
            RDNScorerGymWrapper(env, rdn_scorer), autoencoder, autoencoder_score_norm
        ),
        filter_leaves=filter_leaves,
    )
