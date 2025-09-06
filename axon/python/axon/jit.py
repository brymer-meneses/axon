from typing import Callable, Optional, Any

from .graph_manager import GraphManager
from .axon_bindings import LoweringLevel, Graph, CompilationUnit


class LoweringOps:
    def __init__(self, level: LoweringLevel = LoweringLevel.LLVM, execute: bool = True):
        if level != LoweringLevel.LLVM and execute:
            raise ValueError(
                "Cannot execute operations that are not lowered to LLVM. "
                "Either set level=LoweringLevel.LLVM or execute=False."
            )

        self.execute = execute
        self.level = level


def jit(opts: Optional[LoweringOps] = None) -> Callable:
    def decorator(func: Callable) -> CompiledFunction:
        compiled = CompiledFunction(opts, func)
        compiled.__doc__ = func.__doc__
        compiled.__qualname__ = func.__qualname__
        return compiled

    return decorator


class CompiledFunction:
    def __init__(self, opts: Optional[LoweringOps], func: Callable) -> None:
        self._func: Callable = func
        self._opts: LoweringOps = LoweringOps() if opts is None else opts
        self._cached_graph: Optional[Graph] = None
        self._compiled: Optional[CompilationUnit] = None

    # When `__call__` is invoked every tensor operation is recorded.
    # a new graph is created and checked if it is equal to the
    # current graph.
    def __call__(self, *args, **kwargs) -> Any:
        graph = Graph()
        manager = GraphManager(graph)

        # trace the tensor operations
        with manager:
            manager.trace_params(*args, **kwargs)
            self._func(*args, **kwargs)

        # check if it matches the cached graph
        if graph != self._cached_graph:
            self._compiled = graph.compile(self._opts.level)

        if self._opts.execute:
            assert self._compiled is not None
            return self._compiled.execute(manager.parameters)

    def dump_ir(self) -> str:
        if self._compiled is not None:
            return self._compiled.dump_ir()
        raise RuntimeError("Cannot dump the IR of an uncompiled function.")
