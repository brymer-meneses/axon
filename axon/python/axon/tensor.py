import numpy as np

from . import axon_bindings
from .axon_bindings import ElementType, Tensor

from .graph_manager import GraphManager

def tensor(ndarray, requires_grad=False, element_type=ElementType.Float32) -> Tensor:
    graph = GraphManager.current_graph()
    if graph is not None:
        if requires_grad:
            raise RuntimeError("Cannot set `requires_grad=True` when there is an active graph.")
        return graph.create_constant(ndarray, requires_grad, element_type)

    if not isinstance(ndarray, np.ndarray):
        ndarray = np.array(ndarray, dtype=np.float32)

    return axon_bindings._create_tensor(ndarray, requires_grad, element_type)

