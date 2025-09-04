import contextvars
import typing

from . import axon_bindings

from .axon_bindings import Graph

class GraphManager:
    _current_graph: contextvars.ContextVar[typing.Optional[Graph]] = contextvars.ContextVar("_graph", default=None)

    def __init__(self, graph: Graph) -> None:
        self._token = None
        self._graph = graph

    def __enter__(self, *args, **kwargs) -> "GraphManager":
        self._token = GraphManager._current_graph.set(self._graph)
        axon_bindings._set_current_graph(self._graph)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        if self._token is not None:
            GraphManager._current_graph.reset(self._token)

    @staticmethod
    def current_graph() -> typing.Optional[Graph]:
        graph = GraphManager._current_graph.get() 
        return graph
