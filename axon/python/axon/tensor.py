import numpy as np

from . import axon_bindings as bindings
from .axon_bindings import ElementType

def tensor(ndarray, requires_grad=False, element_type=ElementType.Float32) -> bindings.Tensor | bindings.LazyTensor:
    graph = bindings._get_current_graph()
    if graph is not None:
        if requires_grad:
            raise RuntimeError("Cannot set `requires_grad=True` when there is an active graph.")

        return graph.create_constant(ndarray, requires_grad, element_type)

    if not isinstance(ndarray, np.ndarray):
        ndarray = np.array(ndarray, dtype=np.float32)

    return bindings._create_tensor(ndarray, requires_grad, element_type)

