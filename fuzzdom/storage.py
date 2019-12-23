import torch
import numpy as np
from torch_geometric.data import Batch, Data
from a2c_ppo_acktr.storage import RolloutStorage


class StorageReceipt:
    """
    Convert graphs into integers for easier integration
    """

    def __init__(self):
        self._counter = 0
        self._data = {}

    def __call__(self, state):
        return self.issue_receipt(state)

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, receipt):
        results = [self._data[receipt]]
        return self._wrap_results(results)

    def issue_receipt(self, state):
        assert isinstance(state, Data), str((type(state), state))
        receipt = self._counter
        self._counter += 1
        self._data[receipt] = state
        return torch.tensor([receipt])

    def redeem(self, receipts):
        results = list()
        for idx in receipts.view(-1).tolist():
            results.append(self._data[int(idx)])
        return self._wrap_results(results)

    def _wrap_results(self, results):
        return Batch.from_data_list(results)

    def prune(self, active_receipts):
        keep = set(active_receipts.view(-1).tolist())
        current = set(self._data.keys())
        discard = current - keep
        for d in discard:
            del self._data[d]


class ReceiptRolloutStorage(RolloutStorage):
    def __init__(
        self,
        num_steps,
        num_processes,
        obs_shape,
        action_space,
        recurrent_hidden_state_size,
        receipts,
    ):
        super(ReceiptRolloutStorage, self).__init__(
            num_steps,
            num_processes,
            obs_shape,
            action_space,
            recurrent_hidden_state_size,
        )
        self.receipts = receipts

    def feed_forward_generator(
        self, advantages, num_mini_batch=None, mini_batch_size=None
    ):
        for b in super(ReceiptRolloutStorage, self).feed_forward_generator(
            advantages, num_mini_batch, mini_batch_size
        ):
            obs, *rest = b
            yield (self.receipts.redeem(obs), *rest)

    def recurrent_generator(self, advantages, num_mini_batch):
        for b in super(ReceiptRolloutStorage, self).recurrent_generator(
            advantages, num_mini_batch
        ):
            obs, *rest = b
            yield (self.receipts.redeem(obs), *rest)
