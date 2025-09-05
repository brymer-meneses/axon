import numpy as np
from typing import Tuple, TypeAlias

from . import axon_bindings

from .axon_bindings import dtype, Tensor
from .graph_manager import GraphManager

Shape: TypeAlias = Tuple[int, ...] 

def tensor(ndarray, requires_grad=False, dtype=dtype.float32) -> Tensor:
    graph = GraphManager.current_graph()
    if graph is not None:
        if requires_grad:
            raise RuntimeError("Cannot set `requires_grad=True` when there is an active graph.")
        return graph.create_constant(ndarray, requires_grad, dtype)

    if not isinstance(ndarray, np.ndarray):
        ndarray = np.array(ndarray, dtype=np.float32)

    return axon_bindings._create_tensor(ndarray, requires_grad, dtype)

def ones(shape: Shape, requires_grad=False, dtype=dtype.float32) -> Tensor:
    return axon_bindings._create_filled(shape, 1, requires_grad, dtype)

