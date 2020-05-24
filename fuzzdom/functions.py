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


def init(module, weight_init, bias_init, gain=1):
    if isinstance(gain, str):
        gain = nn.init.calculate_gain(gain)
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


init_c = lambda m, c, nl="relu": init(
    m, lambda x, gain: nn.init.constant_(x, c), lambda x: nn.init.constant_(x, 0),
)
init_ones = lambda m, nl="relu": init(
    m, lambda x, gain: nn.init.ones_(x), lambda x: nn.init.constant_(x, 0),
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
        [s[i, :l].view(l, -1) for i, l in enumerate(x_lens.tolist())], dim=0
    )


def scatter_dev(x, batch, eps=1e-6):
    """
    Compute standard deviation per batch indices
    """
    indices, _counts = torch.unique(batch, sorted=True, return_counts=True)
    _counts = _counts.view(-1, 1)
    _sum = global_add_pool(x, batch)
    assert _counts.shape == _sum.shape, str((_counts.shape, _sum.shape))
    _mean = _sum / _counts
    _idev = (x - _mean[batch]) ** 2 / (_counts[batch] - 1)
    _dev = global_add_pool(_idev, batch) ** 0.5
    return _dev
