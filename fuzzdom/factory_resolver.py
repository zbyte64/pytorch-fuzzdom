import torch
from inspect import signature
from tensorboardX import SummaryWriter


class FactoryResolver(dict):
    """
    Factory resolver
    """

    def __init__(self, cls, **initial_values):
        super().__init__(resolver=self)
        self.update(initial_values)
        self.resolves = cls
        self._resolving = set()

    def get_resolves_attr(self, key):
        if isinstance(self.resolves, dict):
            return self.resolves[key]
        else:
            return getattr(self.resolves, key)

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        assert key not in self._resolving
        self._resolving.add(key)
        f_or_value = self.get_resolves_attr(key)
        if callable(f_or_value):
            value = self(f_or_value)
        else:
            value = f_or_value
        self[key] = value
        self._resolving.remove(key)
        return value

    def report_values(self, writer: SummaryWriter, step_number: int, prefix: str = ""):
        for k, t in self.items():
            if k.startswith("_") or isinstance(t, FactoryResolver):
                continue
            if isinstance(t, torch.Tensor):
                if not len(t.shape):
                    writer.add_scalar(f"{prefix}{k}", t, step_number)
                else:
                    writer.add_histogram(f"{prefix}{k}", t, step_number)
            elif isinstance(t, (int, float)):
                writer.add_scalar(f"{prefix}{k}", t, step_number)
            # ??
            # if hasattr(t, "report_values"):
            #    t.report_values(writer, step_number, f"{prefix}{k}_")

    def __call__(self, func):
        """
        Call a method whose arguments are tensors resolved by matching their name to a self method
        """
        if isinstance(func, str):
            func = self.get_resolves_attr(func)
        deps = signature(func)

        # collect dependencies
        kwargs = {}
        for param in deps.parameters:
            if param == "self":
                continue
            kwargs[param] = self[param]
        return func(**kwargs)
