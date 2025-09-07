import contextvars
import typing

from . import _core

from typing import List, Optional
from ._core import Graph, Tensor


class GraphManager:
    _current_graph: contextvars.ContextVar[Optional[Graph]] = contextvars.ContextVar(
        "_graph", default=None
    )

    def __init__(self, graph: Graph = Graph()) -> None:
        self._token = None
        self._graph = graph
        self._parameters = []

    def __enter__(self, *args, **kwargs) -> "GraphManager":
        self._token = GraphManager._current_graph.set(self._graph)
        _core._set_current_graph(self._graph)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        for param in self._parameters:
            self._graph.untrace(param)

        if self._token is not None:
            GraphManager._current_graph.reset(self._token)

    def _trace(self, tensor: Tensor) -> None:
        self._graph.trace(tensor)
        self._parameters.append(tensor)

    @property
    def parameters(self) -> List[Tensor]:
        return self._parameters

    def trace_params(self, *args, **kwargs) -> None:
        for arg in args:
            if isinstance(arg, Tensor):
                self._trace(arg)
        for _, value in kwargs.items():
            if isinstance(value, Tensor):
                self._trace(value)

    @staticmethod
    def current_graph() -> typing.Optional[Graph]:
        graph = GraphManager._current_graph.get()
        return graph
