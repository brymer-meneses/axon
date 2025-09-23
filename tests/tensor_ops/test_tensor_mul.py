import axon
import torch

from axon import Tensor


def test_mul():
    b1 = torch.randn(10, 10, requires_grad=True)
    b2 = torch.randn(10, 10, requires_grad=True)

    a1 = Tensor(b1, requires_grad=True)
    a2 = Tensor(b2, requires_grad=True)

    a3 = a1 * a2
    grad = Tensor.ones((10, 10))
    a3.backward(grad)

    b_mul = b1 * b2
    b_mul.backward(torch.ones_like(b1))

    axon.testing.assert_are_close(a1.grad, b1.grad)
    axon.testing.assert_are_close(a2.grad, b2.grad)
    axon.testing.assert_are_close(a3, b_mul)


def test_broadcast_mul():
    b1 = torch.randn(10, 10, requires_grad=True)
    b2 = torch.randn(1, 10, requires_grad=True)

    a1 = Tensor(b1, requires_grad=True)  # Shape: (10, 10)
    a2 = Tensor(b2, requires_grad=True)  # Shape: (1, 10)
    a3 = a1 * a2
    grad = Tensor.ones((10, 10))
    a3.backward(grad)

    b_mul = b1 * b2
    b_mul.backward(torch.ones_like(b1))

    axon.testing.assert_are_close(a1.grad, b1.grad)
    axon.testing.assert_are_close(a2.grad, b2.grad)
    axon.testing.assert_are_close(a3, b_mul)
