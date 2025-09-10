import axon
import numpy as np


def test_mul():
    @axon.jit()
    def mul(a, b):
        grad = axon.ones((10, 10))
        l = a * b
        l.backward(grad)

    b1 = np.random.rand(10, 10).astype(np.float32)
    b2 = np.random.rand(10, 10).astype(np.float32)

    a1 = axon.tensor(b1, requires_grad=True, dtype=axon.float32)
    a2 = axon.tensor(b2, requires_grad=True, dtype=axon.float32)

    mul(a1, a2)

    assert axon.testing.is_equal(a1.grad, b2)
    assert axon.testing.is_equal(a2.grad, b1)


def test_broadcast_mul():
    @axon.jit()
    def broadcast_mul(a, b):
        grad = axon.ones((10, 10))
        l = a * b
        l.backward(grad)

    b1 = np.random.rand(10, 10).astype(np.float32)  # Shape: (10, 10)
    b2 = np.random.rand(1, 10).astype(np.float32)  # Shape: (1, 10)

    a1 = axon.tensor(b1, requires_grad=True, dtype=axon.float32)  # Shape: (10, 10)
    a2 = axon.tensor(b2, requires_grad=True, dtype=axon.float32)  # Shape: (1, 10)

    broadcast_mul(a1, a2)

    # For multiplication: dl/da1 = a2 (broadcasted to (10, 10))
    # For multiplication: dl/da2 = a1 (summed to reduce back to (1, 10))

    # a1.grad should be a2 broadcasted to (10, 10)
    # we need the copy here since nanobind cannot take this as an input :<
    expected_a1_grad = np.broadcast_to(b2, (10, 10)).copy()
    assert axon.testing.is_equal(a1.grad, expected_a1_grad)

    # a2.grad should be a1 summed along axis 0 to get shape (1, 10)
    expected_a2_grad = np.sum(b1, axis=0, keepdims=True)
    assert axon.testing.is_equal(a2.grad, expected_a2_grad)
