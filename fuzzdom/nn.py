import torch
from torch import nn
from torch_geometric.nn import APPNP
import torch.nn.functional as F

from a2c_ppo_acktr.utils import init

from .functions import *


class EdgeAttrs(nn.Module):
    """
    Generate edge attributes from node pairs
    """

    def __init__(self, input_dim, out_dim, prior_edge_size):
        super(EdgeAttrs, self).__init__()
        self.edge_attr_fn1 = nn.Sequential(
            init_xn(nn.Linear(input_dim, input_dim), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(input_dim, out_dim), "relu"),
            nn.ReLU(),
        )
        self.edge_attr_fn2 = nn.Sequential(
            init_xn(nn.Linear(input_dim, input_dim), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(input_dim, out_dim), "relu"),
            nn.ReLU(),
        )
        self.edge_attr_fn3 = nn.Sequential(
            init_xn(nn.Linear(input_dim, input_dim), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(input_dim, out_dim), "relu"),
            nn.ReLU(),
        )
        self.edge_attr_fn4 = nn.Sequential(
            init_xn(nn.Linear(input_dim, input_dim), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(input_dim, out_dim), "relu"),
            nn.ReLU(),
        )
        self.edge_sim = nn.CosineSimilarity()
        self.edge_fn = nn.Sequential(
            init_xn(nn.Linear(1 + 4 * out_dim + prior_edge_size, out_dim), "tanh"),
            nn.Tanh(),
        )

    def forward(self, x, edge_index, edge_attr):
        x_src = x[edge_index[0]]
        x_dst = x[edge_index[1]]
        y = torch.cat(
            [
                self.edge_attr_fn1(x_src),
                self.edge_attr_fn2(x_dst),
                self.edge_attr_fn3(x_src - x_dst),
                self.edge_attr_fn4(x_src * x_dst),
            ],
            dim=1,
        )
        s = self.edge_sim(x_src.unsqueeze(2), x_dst.unsqueeze(2))
        z = torch.cat([y, s, edge_attr], dim=1)
        return self.edge_fn(z)


class EdgeMask(nn.Module):
    """
    Propagate a mask [0,1] along edges with attributes
    """

    def __init__(self, edge_dim, mask_dim=1, K=5, bias=True):
        super(EdgeMask, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1e-3))
        self.edge_fn = nn.Sequential(
            init_xn(nn.Linear(edge_dim, 1), "sigmoid"), nn.Sigmoid()
        )
        self.conv = APPNP(K=K, alpha=self.alpha)
        self.K = K
        if bias:
            self.bias = nn.Parameter(torch.ones(mask_dim) * -3)
        else:
            sekf.bias = None

    def forward(self, edge_attr, mask, edge_index):
        edge_weights = self.edge_fn(edge_attr).view(-1)
        # assert False, str((edge_attr.shape, mask.shape, edge_index.shape))
        fill = torch.relu(mask)
        fill = self.conv(fill, edge_index, edge_weights)
        if self.bias is not None:
            fill = fill - F.softplus(self.bias)
        fill = torch.tanh(fill)
        return fill, edge_weights
