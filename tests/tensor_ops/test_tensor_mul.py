import axon
import numpy as np

from axon import Tensor


def test_mul():
    b1 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 10)

    a1 = Tensor(b1, requires_grad=True)
    a2 = Tensor(b2, requires_grad=True)

    a3 = a1 * a2
    a3.backward(Tensor.ones((10, 10)))

    axon.testing.assert_are_close(a1.grad, b2)
    axon.testing.assert_are_close(a2.grad, b1)
    axon.testing.assert_are_close(a3, b1 * b2)


def test_broadcast_mul():
    b1 = np.random.rand(10, 10)
    b2 = np.random.rand(1, 10)

    a1 = Tensor(b1, requires_grad=True)  # Shape: (10, 10)
    a2 = Tensor(b2, requires_grad=True)  # Shape: (1, 10)
    a3 = a1 * a2
    a3.backward(Tensor.ones((10, 10)))

    # For multiplication: dl/da1 = a2 (broadcasted to (10, 10))
    # For multiplication: dl/da2 = a1 (summed to reduce back to (1, 10))

    # a1.grad should be a2 broadcasted to (10, 10)
    # we need the copy here since nanobind cannot take this as an input :<
    expected_a1_grad = np.broadcast_to(b2, (10, 10)).copy()
    axon.testing.assert_are_close(a1.grad, expected_a1_grad)

    # a2.grad should be a1 summed along axis 0 to get shape (1, 10)
    expected_a2_grad = np.sum(b1, axis=0, keepdims=True)
    axon.testing.assert_are_close(a2.grad, expected_a2_grad)

    axon.testing.assert_are_close(a3, b1 * b2)
