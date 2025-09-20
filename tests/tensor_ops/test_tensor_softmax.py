import axon
from axon._core import LoweringLevel
import numpy as np

from axon import Tensor


def softmax(x, axis, keepdims):
    shifted_x = x - np.max(x, axis=axis)
    e_x = np.exp(shifted_x)
    return e_x / np.sum(e_x, axis=axis, keepdims=keepdims)


def test_softmax_keepdims():
    b = np.random.randn(10, 10)
    t = Tensor(b, requires_grad=True)

    result = t.softmax(0, keepdims=True)
    axon.inspect_ir(result, LoweringLevel.LLVM)

    # axon.testing.assert_are_close(result, softmax(b, 0, keepdims=False))
    # axon.testing.assert_are_close(t.grad, np.ones_like(b))


def test_softmax():
    b = np.random.randn(10, 10)
    t = Tensor(b, requires_grad=True)

    # result = t.softmax(0, keepdims=False)
    # result.backward(Tensor.ones((10,)))

    # axon.testing.assert_are_close(result, b.sum(0))
    # axon.testing.assert_are_close(t.grad, np.ones_like(b))
