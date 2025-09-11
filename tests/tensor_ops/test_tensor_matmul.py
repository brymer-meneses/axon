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

    axon.testing.assert_are_close(a1.grad, expected_a1_grad.copy())
    axon.testing.assert_are_close(a2.grad, expected_a2_grad.copy())
    axon.testing.assert_are_close(a3, b1 @ b2)


def test_matmul_broadcast():
    @axon.jit()
    def matmul_broadcast(a, b):
        grad = axon.ones((3, 10, 10))  # Match output shape
        l = a @ b
        l.backward(grad)
        return l

    b1 = np.random.rand(3, 10, 20).astype(np.float32)
    b2 = np.random.rand(20, 10).astype(np.float32)
    a1 = axon.tensor(b1, requires_grad=True, dtype=axon.float32)
    a2 = axon.tensor(b2, requires_grad=True, dtype=axon.float32)

    result = matmul_broadcast(a1, a2)

    # Expected gradients for broadcasting
    grad_shape = (3, 10, 10)  # Output shape
    grad = np.ones(grad_shape, dtype=np.float32)

    # a1.grad: grad @ b2.T for each batch
    expected_a1_grad = grad @ b2.T  # (3,10,10) @ (10,20) = (3,10,20)

    # a2.grad: sum over batch dimension of b1.T @ grad
    expected_a2_grad = np.sum(b1.transpose(0, 2, 1) @ grad, axis=0)  # Sum over batch

    axon.testing.assert_are_close(a1.grad, expected_a1_grad)
    axon.testing.assert_are_close(a2.grad, expected_a2_grad)
    axon.testing.assert_are_close(result, b1 @ b2)


def test_matmul_broadcast_different_batch():
    @axon.jit()
    def matmul_broadcast(a, b):
        grad = axon.ones((4, 15, 15))
        l = a @ b
        l.backward(grad)
        return l

    b1 = np.random.rand(1, 15, 10).astype(np.float32)
    b2 = np.random.rand(4, 10, 15).astype(np.float32)
    a1 = axon.tensor(b1, requires_grad=True, dtype=axon.float32)
    a2 = axon.tensor(b2, requires_grad=True, dtype=axon.float32)

    result = matmul_broadcast(a1, a2)

    # Output shape: (4, 15, 15)
    grad = np.ones((4, 15, 15), dtype=np.float32)

    # a1.grad: grad @ b2.transpose(0, 2, 1), then sum over broadcasted batch dim
    temp_a1_grad = grad @ b2.transpose(0, 2, 1)  # (4,15,15) @ (4,15,25) = (4,15,25)
    expected_a1_grad = np.sum(temp_a1_grad, axis=0, keepdims=True)  # Sum to (1,15,25)

    # a2.grad: b1.transpose(0, 2, 1) @ grad
    expected_a2_grad = b1.transpose(0, 2, 1) @ grad  # (1,25,15) @ (4,15,15) = (4,25,15)

    axon.testing.assert_are_close(a1.grad, expected_a1_grad)
    axon.testing.assert_are_close(a2.grad, expected_a2_grad)
    axon.testing.assert_are_close(result, b1 @ b2)
