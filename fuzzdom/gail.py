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

from .models import WoBGraphEncoder


class Discriminator(nn.Module):
    """
    Modified GAIL Discriminator to handle graph state, and composite actions
    """

    def __init__(self, input_dim, hidden_dim, device):
        super(Discriminator, self).__init__()

        self.device = device

        self.encoder = WoBGraphEncoder(hidden_dim=hidden_dim)
        self.node_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )
        self.trunk = nn.Sequential(nn.Linear(hidden_dim, 1))

        self.train()
        self.to(device)

        self.optimizer = torch.optim.Adam(self.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def forward(self, inputs, actions):
        x = self.encoder(inputs)
        node_actions = actions[inputs.batch]
        node_votes = inputs.votes
        x = torch.cat([x, node_actions, node_votes], dim=1)
        x = self.node_transform(x)
        return self.trunk(x)

    def compute_grad_pen(
        self, expert_state, expert_action, policy_state, policy_action, lambda_=10
    ):
        alpha = torch.rand(expert_action.size(0), 1)
        alpha = alpha.expand_as(expert_action).to(expert_action.device)

        # CONSIDER: how do we blend graphs of different sizes?
        # if we squash nodes, actions make little sense
        # we can union graphs, actions are relative probs ie concat to match node length
        mixup_action = alpha * expert_action[:, 0] + (1 - alpha) * policy_action[:, 0]
        mixup_action.requires_grad = True

        # foreach batch combine network?
        batch_size = expert_state.batch.max().item() + 1
        mixup_state = []
        for i in range(batch_size):
            _em = expert_state.batch == i
            _pm = policy_state.batch == i
            data = {}
            for key in expert_state:
                if key in ("edge_index", "edge_attr"):
                    continue
                data[key] = torch.stack(
                    [export_state[key][_em], policy_state[key][_pm]]
                )
            data["votes"] = torch.stack([expert_action[_em, 1], policy_action[_pm, 1]])
            data = Data.from_dict(data)
            expert_graph_size = _em.sum().item()
            data.edge_index = torch.stack(
                [
                    expert_action.edge_index[_em],
                    policy_action.edge_index[_pm] + expert_graph_size,
                ]
            )
            mixup_state.append(data)
        mixup_state = Batch.from_data_list(mixup_state)

        disc = self.forward(mixup_state, mixup_action)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=(mixup_state, mixup_action),
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size
        )

        print(policy_data_generator)
        assert len(expert_loader)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader, policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            # policy_state.votes = poli
            print("policy_batch", policy_batch)
            print("expert_batch", expert_batch)
            policy_d = self.forward(policy_state, policy_action[:, 0])

            expert_state, expert_action = expert_batch
            # expert_state = obsfilt(expert_state.numpy(), update=False)
            # expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.forward(expert_state, expert_action[:, 0])

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
