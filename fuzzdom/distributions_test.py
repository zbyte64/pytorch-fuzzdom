import pytest
import os
import logging
import torch
from .distributions import MixedDistribution, NodeObjective, FixedCategorical

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def mixed_dist_factory():
    return MixedDistribution(
        [
            FixedCategorical(torch.arange(0, 10, dtype=torch.float)),
            FixedCategorical(torch.arange(0, 2, dtype=torch.float)),
            FixedCategorical(torch.arange(0, 7, dtype=torch.float)),
            FixedCategorical(torch.arange(7, 11, dtype=torch.float)),
        ]
    )


def reference_dist_factory():
    return FixedCategorical(torch.arange(0, 4 * 5, dtype=torch.float).view(4, 5))


def test_mixed_dist_mode():
    dist = mixed_dist_factory()
    ref_dist = reference_dist_factory()
    modes = dist.mode()
    ref_modes = ref_dist.mode()
    assert modes.shape == ref_modes.shape
    assert tuple(modes.shape) == (4, 1)


def test_mixed_dist_sample():
    dist = mixed_dist_factory()
    ref_dist = reference_dist_factory()
    t = dist.sample()
    ref_t = ref_dist.sample()
    assert t.shape == ref_t.shape


def test_mixed_dist_entropy():
    dist = mixed_dist_factory()
    ref_dist = reference_dist_factory()
    t = dist.entropy()
    ref_t = ref_dist.entropy()
    assert t.shape == ref_t.shape


def test_mixed_dist_log_probs():
    dist = mixed_dist_factory()
    ref_dist = reference_dist_factory()
    t = dist.log_probs(torch.arange(0, 4))
    ref_t = ref_dist.log_probs(torch.arange(0, 4))
    assert t.shape == ref_t.shape


def test_node_objective_mode():
    x = torch.zeros((10,), dtype=torch.float)
    x[1] = 9000.0  # second option
    x[2] = 9000.0  # first option
    x[9] = 9000.0  # last option
    x = x.view(-1, 1)
    batch = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2], dtype=torch.int64)
    dist = NodeObjective().forward((x, batch))
    mode = dist.mode()
    truth = torch.tensor([[1], [0], [4]])
    assert (mode == truth).min().item() == 1, str(mode)
    sample = dist.sample()
    assert sample.shape == truth.shape
    assert (sample == truth).min().item() == 1, str(sample)
    assert dist.log_probs(mode).min().item() >= 0
