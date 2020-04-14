from a2c_ppo_acktr.distributions import Categorical, FixedCategorical
from torch import nn
import torch


class MixedDistribution:
    # a category with mixed episodic sizes
    # children are batch instances
    def __init__(self, children):
        self.children = children

    def _f(self, f):
        """
        Apply a function to each child and cat the results
        """
        r = []
        for i, child in enumerate(self.children):
            r.append(f(child, i))
        return r

    def log_probs(self, actions):
        p = self._f(lambda child, i: child.log_probs(actions[i].view(1, -1)))
        return torch.cat(p, dim=0)

    def entropy(self):
        e = self._f(lambda child, i: child.entropy())
        return torch.stack(e)

    def mode(self):
        m = self._f(lambda child, i: child.mode())
        return torch.cat(m, dim=0).view(-1, 1)

    def sample(self):
        s = self._f(lambda child, i: child.sample())
        return torch.cat(s, dim=0).view(-1, 1)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.children})"


class NodeObjective(nn.Module):
    def forward(self, x, batch):
        n_dist = []
        assert len(batch.shape) == 1
        assert len(x.shape) == 2
        assert batch.shape[0] == x.shape[0]
        max_batch_id = batch.max().item()
        for batch_id in range(max_batch_id + 1):
            # 1 x N
            batch_mask = batch == batch_id
            ni = x[batch_mask].view(1, -1)
            ni_dist = FixedCategorical(logits=ni)
            n_dist.append(ni_dist)
        n_dist = MixedDistribution(n_dist)
        return n_dist
