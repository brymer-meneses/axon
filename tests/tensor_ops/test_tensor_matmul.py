import axon
import numpy as np


def test_matmul():
    @axon.jit()
    def matmul(a, b):
        grad = axon.ones((10, 10))
        l = a @ b
        l.backward(grad)
        return l

    b1 = np.random.rand(10, 5).astype(np.float32)
    b2 = np.random.rand(5, 10).astype(np.float32)

    a1 = axon.tensor(b1, requires_grad=True, dtype=axon.float32)
    a2 = axon.tensor(b2, requires_grad=True, dtype=axon.float32)

    a3 = matmul(a1, a2)

    grad = np.ones((10, 10), dtype=np.float32)
    expected_a1_grad = grad @ b2.T
    expected_a2_grad = b1.T @ grad

    assert axon.testing.is_close(a1.grad, expected_a1_grad.copy())
    assert axon.testing.is_close(a2.grad, expected_a2_grad.copy())
    assert axon.testing.is_close(a3, b1 @ b2)
