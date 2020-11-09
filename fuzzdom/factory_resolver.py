import torch
from inspect import signature
from tensorboardX import SummaryWriter

# reporting hints
writer = None


class FactoryResolver:
    """
    Factory resolver/injector
    """

    _last_resolve = None
    _factory_stack = []
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

    def __enter__(self):
        FactoryResolver._factory_stack.append(self)
        self._suffix = FactoryResolver._last_resolve
        return self

    def __exit__(self, type, value, tb):
        # close factory for further use
        self.initial_values = None

        if self.writer:
            prefix = "_".join(
                filter(bool, map(lambda s: s._suffix, FactoryResolver._factory_stack))
            )
            self.report_values(self.writer, self.step_number, prefix)
        self.state = None
        assert self == FactoryResolver._factory_stack.pop()
        # print("Done", FactoryResolver._factory_stack.pop())

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
        FactoryResolver._last_resolve = key
        f_or_value = self.factories[key]
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
        c = FactoryResolver(outer)  # , **self.state)
        c.state = self.state
        for key, value in self.factories.items():
            if key not in c.factories:
                c.factories[key] = value
        return c

    def merge(self, outer):
        m = self.chain(outer)
        m.state = self.state
        return m
