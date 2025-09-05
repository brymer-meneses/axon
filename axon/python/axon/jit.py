import typing
import contextlib
import contextvars
import itertools

import functools

from dataclasses import dataclass

from . import axon_bindings

from typing import Callable, Optional

from .graph_manager import GraphManager

from .axon_bindings import LoweringLevel, Graph, Tensor

@dataclass
class LoweringOps:
    level: LoweringLevel = LoweringLevel.LLVM
    execute: bool = True

def jit(opts: Optional[LoweringOps] = None) -> typing.Callable:
    def decorator(func: typing.Callable) -> CompiledFunction:
        compiled = CompiledFunction(opts, func)
        compiled.__doc__ = func.__doc__
        compiled.__qualname__ = func.__qualname__
        return compiled
    return decorator

class CompiledFunction:
    def __init__(self, opts: Optional[LoweringOps], func: Callable) -> None:
        self._func = func
        self._opts = LoweringOps() if opts is None else opts
        self._cached_graph = None
        self._compiled = None

    # When `__call__` is invoked every tensor operation is recorded.
    # a new graph is created and checked if it is equal to the 
    # current graph.
    def __call__(self, *args, **kwargs) -> typing.Any:
        graph = Graph()

        # trace the tensor operations
        with GraphManager(graph):
            tensors = _process_params(graph, *args, **kwargs)
            axon_bindings._set_current_graph(graph)
            result = self._func(*args, **kwargs)

        # check if it matches the cached graph
        if graph != self._cached_graph:
            self._compiled = graph.compile(self._opts.level)

        if self._opts.execute and self._opts.level == LoweringLevel.LLVM:
            return self._compiled.execute(tensors)

    def dump_ir(self) -> str:
        return self._compiled.dump_ir()

def _process_params(graph, *args, **kwargs):
    tensors = []
    for arg in args:
        if isinstance(arg, Tensor):
            graph.trace(arg)
            tensors.append(arg)
    for (key, value) in kwargs.items():
        if isinstance(value, Tensor):
            graph.trace(value)
            tensors.append(value)
    return tensors

