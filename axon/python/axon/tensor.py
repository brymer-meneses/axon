import numpy as np
import contextlib
import contextvars

from . import axon_bindings as bindings

from bindings import Tensor, LazyTensor
from .compiler import _current_graph

def tensor(ndarray, requires_grad=False) -> Tensor | LazyTensor:
    graph = _current_graph.get()
    if graph is not None:
        return graph.create_constant(ndarray)

    if not isinstance(ndarray, np.ndarray):
        ndarray = np.array(ndarray)
    return bindings.create_tensor(ndarray, requires_grad)

def rand(shape, requires_grad=False) -> Tensor | LazyTensor:
    return bindings.create_rand(shape, requires_grad)
