import axon
import torch

from axon import Tensor


def test_matmul():
    b1 = torch.randn(10, 5)
    b2 = torch.randn(5, 10)

    a1 = Tensor(b1, requires_grad=True)
    a2 = Tensor(b2, requires_grad=True)

    a3 = a1 @ a2
    a3.backward(Tensor.ones((10, 10)))

    grad = torch.ones((10, 10))
    expected_a1_grad = grad @ b2.T
    expected_a2_grad = b1.T @ grad

    axon.testing.assert_are_close(a1.grad, expected_a1_grad)
    axon.testing.assert_are_close(a2.grad, expected_a2_grad)
    axon.testing.assert_are_close(a3, b1 @ b2)


def test_matmul_broadcast():
    b1 = torch.randn(3, 10, 20)
    b2 = torch.randn(20, 10)
    a1 = Tensor(b1, requires_grad=True)
    a2 = Tensor(b2, requires_grad=True)

    l = a1 @ a2
    l.backward(Tensor.ones((3, 10, 10)))

    # Expected gradients for broadcasting
    grad_shape = (3, 10, 10)  # Output shape
    grad = torch.ones(grad_shape)

    # a1.grad: grad @ b2.T for each batch
    expected_a1_grad = grad @ b2.T  # (3,10,10) @ (10,20) = (3,10,20)

    # a2.grad: sum over batch dimension of b1.T @ grad
    expected_a2_grad = torch.sum(b1.transpose(1, 2) @ grad, axis=0)  # Sum over batch

    axon.testing.assert_are_close(a1.grad, expected_a1_grad)
    axon.testing.assert_are_close(a2.grad, expected_a2_grad)
    axon.testing.assert_are_close(l, b1 @ b2)


def test_matmul_broadcast_different_batch():
    b1 = torch.randn(1, 15, 10)
    b2 = torch.randn(4, 10, 15)
    a1 = Tensor(b1, requires_grad=True)
    a2 = Tensor(b2, requires_grad=True)
    grad = Tensor.ones((4, 15, 15))

    result = a1 @ a2
    result.backward(grad)

    # Output shape: (4, 15, 15)
    grad = torch.ones((4, 15, 15))

    temp_a1_grad = grad @ b2.transpose(1, 2)  # (4,15,15) @ (4,15,25) = (4,15,25)
    expected_a1_grad = torch.sum(
        temp_a1_grad, axis=0, keepdims=True
    )  # Sum to (1,15,25)

    expected_a2_grad = b1.transpose(1, 2) @ grad  # (1,25,15) @ (4,15,15) = (4,25,15)

    axon.testing.assert_are_close(a1.grad, expected_a1_grad)
    axon.testing.assert_are_close(a2.grad, expected_a2_grad)
    axon.testing.assert_are_close(result, b1 @ b2)
