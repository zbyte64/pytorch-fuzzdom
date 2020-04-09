from concurrent.futures.process import ProcessPoolExecutor
import torch.multiprocessing as mp


class PytorchProcessPoolExecutor(ProcessPoolExecutor):
    def __init__(self, *args, **kwargs):
        kwargs["mp_context"] = mp
        ProcessPoolExecutor.__init__(self, *args, **kwargs)
