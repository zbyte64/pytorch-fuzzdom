import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd

from torch_geometric.nn import (
    TopKPooling,
    SAGEConv,
    SAGPooling,
    global_max_pool,
    GlobalAttention,
    GCNConv,
    NNConv,
)
from torch_geometric.data import Data, Batch

from baselines.common.running_mean_std import RunningMeanStd

from .models import WoBObservationEncoder


class Discriminator(nn.Module):
    """
    Modified GAIL Discriminator to handle graph state, and composite actions
    """

    def __init__(self, input_dim, hidden_dim, device):
        super(Discriminator, self).__init__()

        self.device = device

        self.encoder = WoBObservationEncoder(out_dim=hidden_dim)
        self.trunk = nn.Sequential(nn.Linear(hidden_dim, 1))

        self.train()
        self.to(device)

        self.optimizer = torch.optim.Adam(self.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def forward(self, inputs, votes):
        x = self.encoder(inputs, votes)
        return self.trunk(x)

    def compute_grad_pen(
        self, expert_state, expert_action, policy_state, policy_action, lambda_=10
    ):
        # merge graphs, apply alpha to vote shares
        mixup_state = Batch()
        for key, value in expert_state:
            assert isinstance(key, str), str(key)
            if key in ("edge_index", "edge_attr"):
                continue
            mixup_state[key] = torch.cat([expert_state[key], policy_state[key]])
        mixup_state.edge_index = torch.cat(
            [
                expert_state.edge_index,
                policy_state.edge_index + expert_state.batch.shape[0],
            ],
            dim=1,
        )

        alpha = torch.rand(expert_action.size(0))
        batch_size = expert_state.batch.max().item() + 1
        mixup_votes = []
        for i in range(batch_size):
            _em = expert_state.batch == i
            _pm = policy_state.batch == i
            votes = torch.zeros((_em.sum() + _pm.sum()).item())
            assert votes.shape[0]
            votes[expert_action[i]] = alpha[i]
            votes[policy_action[i] + _em.sum().item()] = 1 - alpha[i]
            mixup_votes.append(votes)
        mixup_action = torch.cat(mixup_votes).view(-1, 1)
        mixup_action.requires_grad = True

        disc = self.forward(mixup_state, mixup_action)
        ones = torch.ones(disc.size()).to(disc.device)
        inputs = [mixup_action]
        for key, value in mixup_state:
            if value.dtype == torch.float:
                value.requires_grad = True
                inputs.append(value)
        grad = autograd.grad(
            outputs=disc,
            inputs=inputs,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size
        )
        assert len(expert_loader)

        def _votes(state, action):
            batch_size = state.batch.max().item() + 1
            votes = torch.zeros(state.value.shape[0], 1)
            past = 0
            for b_idx in range(batch_size):
                _m = state.batch == b_idx
                _size = _m.sum().item()
                votes[past + action[b_idx], 0] = 1
                past += _size
            return votes

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader, policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            batch_size = policy_state.batch.max().item() + 1
            policy_votes = _votes(policy_state, policy_action)
            policy_d = self.forward(policy_state, policy_votes)

            expert_state, expert_action, _ = expert_batch
            # expert_state = obsfilt(expert_state.numpy(), update=False)
            # expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_votes = _votes(expert_state, expert_action)
            expert_d = self.forward(expert_state, expert_votes)

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d, torch.ones(expert_d.size()).to(self.device)
            )
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d, torch.zeros(policy_d.size()).to(self.device)
            )

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(
                expert_state, expert_action, policy_state, policy_action
            )

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
