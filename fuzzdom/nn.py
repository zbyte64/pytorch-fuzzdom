from inspect import signature
import torch
from torch import nn
from torch_geometric.nn import APPNP
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from a2c_ppo_acktr.utils import init

from .functions import *


class ResolveMixin:
    def start_resolve(self, info: dict):
        """
        Initializes resolves process with initial values
        """
        self._last_values = info

    def export_value(self, key, value):
        self._last_values[key] = value

    def resolve_value(self, func, resolving=None):
        """
        Call a method whose arguments are tensors resolved by matching their name to a self method
        """
        fname = func.__name__
        deps = signature(func)

        if fname not in self._last_values:
            # collect dependencies
            resolving = resolving or {fname}
            kwargs = {}
            for param in deps.parameters:
                if param == "self":
                    continue
                if param not in self._last_values:
                    assert (
                        param not in resolving
                    ), f"{fname} has circular dependency through {param}"
                    resolving.add(param)
                    assert hasattr(self, param), f"{fname} has unmet dependency {param}"
                    value = self.resolve_value(getattr(self, param), resolving)
                else:
                    value = self._last_values[param]
                kwargs[param] = value
            result = func(**kwargs)
            assert result is not None, f"Bad return value: {fname}"
            if fname == "_lambda_":
                return result
            self._last_values[fname] = result
        return self._last_values[fname]

    def report_values(self, writer: SummaryWriter, step_number: int, prefix: str = ""):
        for k, t in self._last_values.items():
            if k.startswith("_"):
                continue
            if hasattr(t, "cpu"):
                writer.add_histogram(f"{prefix}{k}", t, step_number)
        for k, mod in self.named_children():
            if hasattr(mod, "report_values"):
                mod.report_values(writer, step_number, f"{prefix}{k}_")


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
