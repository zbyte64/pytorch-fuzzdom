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
