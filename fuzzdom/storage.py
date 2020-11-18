import torch
import numpy as np
from torch_geometric.data import Data
import random

from .data import TupleBatch


class StorageReceipt:
    """
    Convert graphs into integers for easier integration
    """

    def __init__(self, device="cpu", storage_device="cpu"):
        self._counter = 0
        self._data = {}
        self.device = device
        self.storage_device = storage_device

    def __call__(self, state):
        return self.issue_receipt(state)

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, receipt):
        results = [self._data[receipt]]
        return self._wrap_results(results)

    def __reduce__(self):
        return (StorageReceipt, tuple())

    def issue_receipt(self, state):
        # always have actions
        assert len(state[2])
        receipt = self._counter
        self._counter += 1
        self._data[receipt] = tuple(map(lambda x: x.to(self.storage_device), state))
        return torch.tensor([receipt])

    def redeem(self, receipts):
        results = list()
        for idx in receipts.view(-1).tolist():
            results.append(self._data[int(idx)])
        return self._wrap_results(results)

    def _wrap_results(self, results):
        return tuple(
            map(lambda x: x.to(self.device), TupleBatch.from_data_list(results))
        )

    def prune(self, active_receipts):
        keep = set(active_receipts.view(-1).tolist())
        current = set(self._data.keys())
        discard = current - keep
        for d in discard:
            del self._data[d]


class RandomizedReplayStorage:
    def __init__(self, identifier, alpha=0.5, storage_device="cpu", device="cpu"):
        self.identifier = identifier
        self.device = device
        self.storage_device = storage_device
        self.alpha = alpha
        self._data = dict()

    def insert(self, states):
        for id, state in zip(map(self.identifier, states), states):
            self._data[id] = tuple(map(lambda x: x.to(self.storage_device), state))

    def next(self):
        sample_size = int(len(self._data) * self.alpha)
        if sample_size < 1:
            return
        ids = random.sample(list(self._data.keys()), sample_size)
        for id in ids:
            yield tuple(map(lambda x: x.to(self.device), self._data.pop(id)))
