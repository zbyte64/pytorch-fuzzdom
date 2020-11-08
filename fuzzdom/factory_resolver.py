import torch
from inspect import signature
from tensorboardX import SummaryWriter


class FactoryResolver:
    """
    Factory resolver/injector
    """

    def __init__(self, factories, **initial_values):
        self.state = initial_values
        if not isinstance(factories, dict):
            self.factories = {k: getattr(factories, k) for k in dir(factories)}
        else:
            self.factories = factories
        self._resolving = set()

    def items(self):
        return self.state.items()

    def update(self, other):
        return self.state.update(other)

    def __iter__(self):
        return iter(self.state)

    def __setitem__(self, key, value):
        self.state[key] = value

    def __hasitem__(self, key):
        return key in self.state

    def __getitem__(self, key):
        if key in self.state:
            return self.state[key]
        assert key not in self._resolving
        self._resolving.add(key)
        f_or_value = self.factories[key]
        if callable(f_or_value):
            value = self(f_or_value)
        else:
            value = f_or_value
        self.state[key] = value
        self._resolving.remove(key)
        return value

    def report_values(self, writer: SummaryWriter, step_number: int, prefix: str = ""):
        for k, t in self.state.items():
            if k.startswith("_") or isinstance(t, FactoryResolver):
                continue
            if isinstance(t, torch.Tensor):
                if not len(t.shape):
                    writer.add_scalar(f"{prefix}{k}", t, step_number)
                else:
                    writer.add_histogram(f"{prefix}{k}", t, step_number)
            elif isinstance(t, (int, float)):
                writer.add_scalar(f"{prefix}{k}", t, step_number)

    def __call__(self, func):
        """
        Call a method whose arguments are tensors resolved by matching their name to a self method
        """
        if isinstance(func, str):
            func = self.factories[func]
        deps = signature(func)

        # collect dependencies
        kwargs = {}
        for param in deps.parameters:
            if param == "self":
                continue
            if param == "resolver":
                kwargs[param] = self
            else:
                kwargs[param] = self[param]
        return func(**kwargs)

    def __reduce__(self):
        # disable direct pickling
        return (type(self), ({},))

    def chain(self, outer):
        c = FactoryResolver(outer, **self.state)
        for key, value in self.factories.items():
            if key not in c.factories:
                c.factories[key] = value
        return c

    def merge(self, outer):
        m = self.chain(outer)
        m.state = self.state
        return m
