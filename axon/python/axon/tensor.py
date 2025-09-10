import numpy as np
from typing import Tuple, TypeAlias

from . import _core
import axon

from ._core import dtype, Tensor
from .graph_manager import GraphManager

Shape: TypeAlias = Tuple[int, ...]


def _as_axon_dtype(dtype: axon.dtype) -> np.dtype:
    if dtype == axon.float32:
        return np.float32
    if dtype == axon.float64:
        return np.float64


def tensor(ndarray, requires_grad=False, dtype=dtype.float32) -> Tensor:
    graph = GraphManager.current_graph()
    if graph is not None:
        if requires_grad:
            raise RuntimeError(
                "Cannot set `requires_grad=True` when there is an active graph."
            )
        return graph.create_constant(ndarray, requires_grad, dtype)

    if not isinstance(ndarray, np.ndarray):
        ndarray = np.array(ndarray)

    np_dtype = _as_axon_dtype(dtype)
    return _core._create_tensor(ndarray.astype(np_dtype), requires_grad, dtype)


def ones(shape: Shape, requires_grad=False, dtype=dtype.float32) -> Tensor:
    return _core._create_filled(shape, 1, requires_grad, dtype)


def zeros(shape: Shape, requires_grad=False, dtype=dtype.float32) -> Tensor:
    return _core._create_filled(shape, 0, requires_grad, dtype)


def randn(shape: Shape, requires_grad=False, dtype=dtype.float32) -> Tensor:
    return _core._create_randn(shape, requires_grad, dtype)
