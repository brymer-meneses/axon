from axon import Tensor
import axon._core as _core


def is_equal(left: Tensor, right) -> bool:
    return _core._is_equal(left, right)


def is_close(left: Tensor, right, tolerance: float = 1e-5) -> bool:
    return _core._is_close(left, right, tolerance)
