import gym
import torch
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
        self.loss_std_dev = torch.tensor(0.01)

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
        return scores / self.loss_std_dev.clamp(0.0001, 1000) / 10000

    def loss(self, dom, logs):
        self.start_resolve({"dom": dom, "logs": logs})
        scores = self.resolve_value(self.raw_score)
        loss = scores.mean()
        with torch.no_grad():
            self.loss_std_dev = scores.std()
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
            print("RDN SCORE", score)
            return score.item()


class AEScorerGymWrapper(gym.Wrapper):
    def __init__(self, env, autoencoder):
        super().__init__(env)
        self.autoencoder = autoencoder

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward += self.score_observation(observation)
        return observation, reward, done, info

    def score_observation(self, observation):
        if not hasattr(self.autoencoder, "loss_std_dev"):
            return 0
        dom = observation[0]
        with torch.no_grad():
            x = autoencoder_x(dom)
            z = self.autoencoder.encode(x, dom.edge_index)
            score = self.autoencoder.recon_loss(z, dom.edge_index)
            score = score / self.autoencoder.loss_std_dev.clamp(0.0001, 1000) / 10000
            print("AE SCORE", score)
            return score.item()


def make_rdn_vec_envs(envs, receipts, rdn_scorer, autoencoder, filter_leaves=None):

    return make_vec_envs(
        envs,
        receipts,
        inner=lambda env: AEScorerGymWrapper(
            RDNScorerGymWrapper(env, rdn_scorer), autoencoder
        ),
        filter_leaves=filter_leaves,
    )
