import numpy as np
import contextlib
import contextvars

from . import axon_bindings as bindings

def tensor(ndarray, requires_grad=False) -> bindings.Tensor | bindings.LazyTensor:
    graph = bindings._get_current_graph()
    if graph is not None:
        return graph.create_constant(ndarray)

    if not isinstance(ndarray, np.ndarray):
        ndarray = np.array(ndarray)
    return bindings._create_tensor(ndarray, requires_grad)

