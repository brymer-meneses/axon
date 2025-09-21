import axon
from axon._core import LoweringLevel
import numpy as np

from axon import Tensor


def softmax(x, axis):
    shifted_x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(shifted_x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def test_softmax():
    b = np.random.randn(100, 100)
    t = Tensor(b, requires_grad=True)

    s0 = t.softmax(0)

    axon.testing.assert_are_close(s0, softmax(b, 0))
