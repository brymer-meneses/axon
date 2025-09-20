import numpy as np

from axon import Tensor
import axon


def test_sum_keepdims():
    b = np.random.randn(10, 10)
    t = Tensor(b, requires_grad=True)

    result = t.sum(0, keepdims=True)
    result.backward(Tensor.ones((1, 10)))

    axon.testing.assert_are_close(result, b.sum(0, keepdims=True))
    axon.testing.assert_are_close(t.grad, np.ones_like(b))


def test_sum():
    b = np.random.randn(10, 10)
    t = Tensor(b, requires_grad=True)

    result = t.sum(0)
    result.backward(Tensor.ones((10,)))

    axon.testing.assert_are_close(result, b.sum(0))
    axon.testing.assert_are_close(t.grad, np.ones_like(b))
