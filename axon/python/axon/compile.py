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
        self._compiled = None

    # When `__call__` is invoked every tensor operation is recorded.
    # a new graph is created and checked if it is equal to the 
    # current graph.
    def __call__(self, *args, **kwargs) -> typing.Any:
        graph = bindings.Graph()
        args, kwargs = _convert_params(graph, *args, **kwargs)

        # trace the tensor operations
        bindings._set_current_graph(graph)
        result = self._func(*args, **kwargs)

        # check if it matches the cached graph
        if graph != self._cached_graph:
            self._compiled = graph.compile()

        return result

    def dump_ir(self) -> str:
        return self._compiled.__repr__()

def _convert_params(graph, *args, **kwargs):
    new_args = []
    for arg in args:
        if isinstance(arg, bindings.Tensor):
            arg = graph.declare_parameter(arg.shape, arg.requires_grad)
        new_args.append(arg)
    new_kwargs = {}
    for (key, value) in kwargs.items():
        if isinstance(value, bindings.Tensor):
            value = graph.declare_parameter(value.shape, value.requires_grad)
        new_kwargs[key] = value
    return tuple(new_args), new_kwargs

def compile(func: typing.Callable) -> typing.Callable:
    compiled = CompiledFunction(func)
    compiled.__doc__ = func.__doc__
    compiled.__qualname__ = func.__qualname__
    return compiled
