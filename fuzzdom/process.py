"""
pytorch friendly ProcessPoolExecutor

Adapted from: https://github.com/python/cpython/blob/master/Lib/concurrent/futures/process.py

Python 3.7!
"""
from concurrent.futures.process import *
from multiprocessing.context import SpawnContext
import torch.multiprocessing as mp
from torch.multiprocessing.queue import Queue


class _PytorchSafeQueue(Queue):
    """Safe Queue set exception to the future object linked to a job"""

    def __init__(self, max_size=0, *, ctx, pending_work_items):
        self.pending_work_items = pending_work_items
        super().__init__(max_size, ctx=ctx)

    def _on_queue_feeder_error(self, e, obj):
        if isinstance(obj, _CallItem):
            tb = traceback.format_exception(type(e), e, e.__traceback__)
            e.__cause__ = _RemoteTraceback('\n"""\n{}"""'.format("".join(tb)))
            work_item = self.pending_work_items.pop(obj.work_id, None)
            # work_item can be None if another process terminated. In this case,
            # the queue_manager_thread fails all work_items with BrokenProcessPool
            if work_item is not None:
                work_item.future.set_exception(e)
        else:
            super()._on_queue_feeder_error(e, obj)


class PytorchProcessPoolExecutor(ProcessPoolExecutor):
    def __init__(self, *args, **kwargs):
        kwargs["mp_context"] = mp  # mp_context
        ProcessPoolExecutor.__init__(self, *args, **kwargs)
        queue_size = self._max_workers + EXTRA_QUEUED_CALLS
        self._call_queue = _PytorchSafeQueue(
            max_size=queue_size,
            ctx=self._mp_context,
            pending_work_items=self._pending_work_items,
        )


class PytorchContext(SpawnContext):
    def SimpleQueue(self):
        return mp.SimpleQueue()

    def Queue(self, maxsize=0):
        return mp.Queue(maxsize, ctx=self.get_context())


mp_context = PytorchContext()
