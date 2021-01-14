import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch_geometric.data import Data
from collections import defaultdict, deque
import random

from .data import TupleBatch


def send_data_to(data: Data, device: str):
    for key in data.keys:
        v = data[key]
        if hasattr(v, "detach"):
            v = v.detach()
        if hasattr(v, "to"):
            v = v.to(device)
        else:
            continue
        data[key] = v
    return data


class StorageReceipt:
    """
    Convert graphs into integers for easier integration
    """

    def __init__(self, device="cpu", storage_device="cpu"):
        self._counter = 0
        self._data = {}
        self.device = device
        self.storage_device = storage_device

    def __call__(self, state: tuple):
        return self.issue_receipt(state)

    def __contains__(self, key: int):
        return key in self._data

    def __getitem__(self, receipt: int):
        results = [self._data[receipt]]
        return self._wrap_results(results)

    def __reduce__(self):
        return (StorageReceipt, tuple())

    def issue_receipt(self, state: tuple):
        # always have actions
        assert len(state[2])
        receipt = self._counter
        self._counter += 1
        self._data[receipt] = tuple(
            (send_data_to(x, self.storage_device) for x in state)
        )
        return torch.tensor([receipt])

    def redeem(self, receipts):
        results = list()
        for idx in receipts.view(-1).tolist():
            results.append(self._data[int(idx)])
        return self._wrap_results(results)

    def _wrap_results(self, results):
        return tuple(
            (send_data_to(c, self.device) for c in TupleBatch.from_data_list(results))
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
            self._data[id] = tuple(
                (send_data_to(x, self.storage_device) for x in state)
            )

    def next(self):
        sample_size = int(len(self._data) * self.alpha)
        if sample_size < 1:
            return
        ids = random.sample(list(self._data.keys()), sample_size)
        for id in ids:
            yield tuple((send_data_to(x, self.device) for x in self._data.pop(id)))


class TaskQueueReplayStorage:
    def __init__(self, batch_size, maxlen, device):
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.device = device
        self.states = defaultdict(lambda: deque(maxlen=maxlen))
        self.actions = defaultdict(lambda: deque(maxlen=maxlen))

    def store_episode(self, task, states, actions):
        assert len(states) == len(actions), str((states, actions))
        self.states[task].extend(states)
        self.actions[task].extend(actions)

    def __len__(self):
        return sum((len(v) for v in self.states.values()))

    def __iter__(self):
        all_states = [frame for v in self.states.values() for frame in v]
        all_actions = [frame for v in self.actions.values() for frame in v]
        sampler = BatchSampler(
            SubsetRandomSampler(range(len(all_states))), self.batch_size, drop_last=True
        )
        for indices in sampler:
            yield tuple(
                (
                    send_data_to(c, self.device)
                    for c in TupleBatch.from_data_list([all_states[i] for i in indices])
                )
            ), torch.tensor([all_actions[i] for i in indices]).to(self.device)
