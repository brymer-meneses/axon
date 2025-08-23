import typing
import contextlib
import contextvars
import itertools

import functools

from . import axon_bindings as bindings

class CompiledFunction:
    def __init__(self, func: typing.Callable) -> None:
        self._func = func
        self._cached_graph = None

    # When `__call__` is invoked every tensor operation is recorded.
    # a new graph is created and checked if it is equal to the 
    # current graph.
    def __call__(self, *args, **kwargs) -> typing.Any:
        graph = bindings.Graph()
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, bindings.Tensor):
                graph.declare_parameter(arg.shape, arg.requires_grad)

        # trace the tensor operations
        bindings._set_current_graph(graph)
        result = self._func(*args, **kwargs)
        graph.finalize(result)

        # check if it matches the cached graph
        if graph == self._cached_graph:
            pass

        return result


def compile(func: typing.Callable) -> typing.Callable:
    @functools.wraps(func)
    def decorated_func(*args, **kwargs) -> typing.Any:
        compiled_func = CompiledFunction(func)
        return compiled_func(*args, **kwargs)
    return decorated_func
