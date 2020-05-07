import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from torch_geometric.utils import add_self_loops
from torch_geometric.nn import (
    global_max_pool,
    global_add_pool,
    global_mean_pool,
)
from torch_scatter import scatter
from a2c_ppo_acktr.utils import init


init_i = lambda m: init(
    m,
    lambda x, gain: nn.init.ones_(x),
    lambda x: nn.init.constant_(x, 0) if x else None,
)
init_xu = lambda m, nl="relu": init(
    m,
    nn.init.xavier_uniform_,
    lambda x: nn.init.constant_(x, 0),
    nn.init.calculate_gain(nl),
)
init_xn = lambda m, nl="relu": init(
    m,
    nn.init.xavier_normal_,
    lambda x: nn.init.constant_(x, 0),
    nn.init.calculate_gain(nl),
)
init_ot = lambda m, nl="relu": init(
    m,
    nn.init.orthogonal_,
    lambda x: nn.init.constant_(x, 0),
    nn.init.calculate_gain(nl),
)
init_ = init_ot
init_r = lambda m: init_ot(m, "relu")
init_t = lambda m: init_ot(m, "tanh")

reverse_edges = lambda e: torch.stack([e[1], e[0]]).contiguous()
full_edges = lambda e: torch.cat(
    [add_self_loops(e)[0], reverse_edges(e)], dim=1
).contiguous()


def global_min_pool(x, batch, size=None):
    size = batch.max().item() + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce="min")


def global_norm_pool(x, batch, eps=1e-6):
    _max = global_max_pool(x, batch)[batch]
    _min = global_min_pool(x, batch)[batch]
    return (x - _min + eps) / (_max - _min + eps)


def pack_as_sequence(x, batch):
    seq = [x[batch == i] for i in range(batch.max().item() + 1)]
    x_lens = [int(s.shape[0]) for s in seq]
    seq_pad = pad_sequence(seq, batch_first=False, padding_value=0.0)
    seq_pac = pack_padded_sequence(
        seq_pad, x_lens, batch_first=False, enforce_sorted=False
    )
    return seq_pac


def unpack_sequence(output_packed):
    s, x_lens = pad_packed_sequence(output_packed, batch_first=True)
    return torch.cat(
        [s[i, :l].view(l) for i, l in enumerate(x_lens.tolist())], dim=0
    ).view(-1, 1)
