from a2c_ppo_acktr.distributions import Categorical, FixedCategorical
from torch import nn
import torch


class MixedDistribution:
    # a category with mixed episodic sizes
    # children are batch instances
    def __init__(self, *children):
        self.children = children

    def _f(self, f):
        """
        Apply a function to each child and cat the results
        """
        r = []
        for i, child in enumerate(self.children):
            r.append(f(child, i).view(-1, 1))
        r = torch.cat(r, dim=1)
        return r

    def log_probs(self, actions):
        return self._f(lambda child, i: child.log_probs(actions[i].view(-1, 1)))

    def entropy(self):
        return self._f(lambda child, i: child.entropy())

    def mode(self):
        return self._f(lambda child, i: child.mode())

    def sample(self):
        return self._f(lambda child, i: child.sample())

    def __repr__(self):
        return f"{self.__class__.__name__}({self.children})"


class CatDistribution(MixedDistribution):
    # batch level mixture of features
    # children are categories
    def log_probs(self, actions):
        p = self._f(lambda child, i: child.log_probs(actions[:, i].view(-1, 1)))
        return p.sum(dim=1, keepdim=True)

    def entropy(self):
        e = self._f(lambda child, i: child.entropy())
        return e.sum(dim=1, keepdim=True)

    def mode(self):
        m = self._f(lambda child, i: child.mode())
        return m
        return m.sum(dim=1, keepdim=True)


class NodeObjective(nn.Module):
    def forward(self, x):
        n_dist = []
        for ni in x:
            # 1 x N
            ni = ni.view(1, -1)
            ni_dist = FixedCategorical(logits=ni)
            n_dist.append(ni_dist)
        n_dist = MixedDistribution(*n_dist)
        return CatDistribution(n_dist)
