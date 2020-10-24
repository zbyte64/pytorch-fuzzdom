import torch
from torch import nn
from torch_geometric.nn import APPNP
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from a2c_ppo_acktr.utils import init

from .functions import *
from .factory_resolver import FactoryResolver


class ResolveMixin:
    def start_resolve(self, params):
        self._last_values = FactoryResolver(self, **params)

    def export_value(self, key, value):
        self._last_values[key] = value

    def resolve_value(self, func):
        key = func.__name__
        if key in self._last_values:
            return self._last_values[key]
        val = self._last_values(func)
        if not key.startswith("_"):
            self._last_values[key] = val
        return val

    def report_values(self, writer: SummaryWriter, step_number: int, prefix: str = ""):
        self._last_values.report_values(writer, step_number, prefix)
        for k, t in self.namedmodules():
            if hasattr(t, "report_values"):
                t.report_values(writer, step_number, f"{prefix}{k}_")


class EdgeAttrs(nn.Module):
    """
    Generate edge attributes from node pairs
    """

    def __init__(self, input_dim, out_dim):
        super(EdgeAttrs, self).__init__()
        self.edge_fn = nn.Sequential(
            init_ot(nn.Linear(input_dim * 3, input_dim), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(input_dim, out_dim), "sigmoid"),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index):
        x_src = x[edge_index[0]]
        x_dst = x[edge_index[1]]
        _x = torch.cat([x_src, x_dst, x_src - x_dst], dim=1)
        return self.edge_fn(_x)


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
