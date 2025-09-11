from axon import Tensor
import axon._core as _core


def assert_are_equal(left: Tensor, right):
    _core._assert_are_close(left, right, 0.0)


def assert_are_close(left: Tensor, right, tolerance: float = 1e-5):
    _core._assert_are_close(left, right, tolerance)
