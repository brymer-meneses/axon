import typing
import contextlib
import contextvars
import itertools

import functools

from . import axon_bindings as bindings

from .axon_bindings import LoweringOps

def jit(opts: typing.Optional[bindings.LoweringOps] = None) -> typing.Callable:
    def decorator(func: typing.Callable) -> typing.Callable:
        compiled = CompiledFunction(opts, func)
        compiled.__doc__ = func.__doc__
        compiled.__qualname__ = func.__qualname__
        return compiled
    return decorator

class CompiledFunction:
    def __init__(self, opts: typing.Optional[bindings.LoweringOps], func: typing.Callable) -> None:
        self._func = func
        self._opts = LoweringOps(LoweringOps.Level.Axon) if opts is None else opts
        self._cached_graph = None
        self._compiled = None

    # When `__call__` is invoked every tensor operation is recorded.
    # a new graph is created and checked if it is equal to the 
    # current graph.
    def __call__(self, *args, **kwargs) -> typing.Any:
        graph = bindings.Graph()
        traced_args, traced_kwargs = _convert_params(graph, *args, **kwargs)

        # trace the tensor operations
        bindings._set_current_graph(graph)
        result = self._func(*traced_args, **traced_kwargs)

        # check if it matches the cached graph
        if graph != self._cached_graph:
            self._compiled = graph.compile(self._opts)

        return result

    def dump_ir(self) -> str:
        return self._compiled.dump_ir()

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

