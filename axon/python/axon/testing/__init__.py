import axon._core

from axon import Tensor
from numpy.typing import ArrayLike


def assert_are_close(tensor: Tensor, array: ArrayLike):
    axon._core.assert_are_close(tensor, array)
