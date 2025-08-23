import numpy as np
import contextlib
import contextvars

from . import axon_bindings

from .axon_bindings import Tensor, LazyTensor

def tensor(ndarray, requires_grad=False) -> Tensor | LazyTensor:
    graph = axon_bindings._get_current_graph()
    if graph is not None:
        return graph._create_constant(ndarray)

    if not isinstance(ndarray, np.ndarray):
        ndarray = np.array(ndarray)
    return axon_bindings._create_tensor(ndarray, requires_grad)

