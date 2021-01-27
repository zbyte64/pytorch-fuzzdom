import torch
import numpy as np
from inspect import signature
from tensorboardX import SummaryWriter

# reporting hints
writer = None


class FactoryResolver:
    """
    Factory resolver/injector
    """

    _last_resolve = None
    _suffix = None
    # Class Globals
    _factory_stack = []
    _reported_values = set()
    _last_step_number = 0
    step_number = None
    writer = None

    def __init__(self, factories, **initial_values):
        self.initial_values = initial_values
        self.state = dict()
        if not isinstance(factories, dict):
            self.factories = {k: getattr(factories, k) for k in dir(factories)}
        else:
            self.factories = factories
        self._resolving = set()

    def items(self):
        return self.state.items()

    def update(self, other):
        self.state.update(other)

    @classmethod
    def enable_reporting(cls, writer, step_number):
        cls.writer = writer
        cls.step_number = step_number

    @classmethod
    def disable_reporting(cls):
        cls.writer = None

    def __enter__(self):
        self._suffix = None
        if FactoryResolver._factory_stack:
            parent = FactoryResolver._factory_stack[-1]
            if parent._last_resolve:
                self._suffix = parent._last_resolve
        FactoryResolver._factory_stack.append(self)
        return self

    def __exit__(self, type, value, tb):
        # close factory for further use
        self.initial_values = None

        if FactoryResolver.writer:
            self.report_values()
        assert self == FactoryResolver._factory_stack.pop()
        self.state = None

    def __iter__(self):
        return iter(self.state)

    def __setitem__(self, key, value):
        self.state[key] = value

    def __hasitem__(self, key):
        return key in self.state or key in self.initial_values

    def __getitem__(self, key):
        if key in self.state:
            return self.state[key]
        if key in self.initial_values:
            return self.initial_values[key]
        assert key not in self._resolving
        self._resolving.add(key)
        self._last_resolve = key
        f_or_value = self.factories[key]
        if callable(f_or_value):
            value = self(f_or_value)
        else:
            value = f_or_value
        self[key] = value
        self._resolving.remove(key)
        return value

    def report_values(self):
        if FactoryResolver.step_number > FactoryResolver._last_step_number:
            FactoryResolver._last_step_number = FactoryResolver.step_number
            FactoryResolver._reported_values.clear()
        prefix = "_".join(
            filter(bool, map(lambda s: s._suffix, FactoryResolver._factory_stack))
        )
        if prefix:
            prefix += "_"
        # CONSIDER: maybe _reported_values should track individual keys for incremental updates?
        if prefix not in FactoryResolver._reported_values:
            self._report_values(
                FactoryResolver.writer, FactoryResolver.step_number, prefix
            )
            FactoryResolver._reported_values.add(prefix)

    def _report_values(self, writer: SummaryWriter, step_number: int, prefix: str = ""):
        for k, t in self.items():
            if k.startswith("_"):
                continue
            if isinstance(t, (torch.Tensor, np.ndarray)):
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

    def chain(self, _outer=None, **kwargs):
        outer = _outer if _outer is not None else self.factories
        initial_state = dict(self.state)
        initial_state.update(kwargs)
        c = FactoryResolver(outer, **initial_state)
        if _outer is not None:
            for key, value in self.factories.items():
                if key not in c.factories:
                    c.factories[key] = value
        return c
