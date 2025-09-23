import axon
import torch

from axon import Tensor


def test_sub():
    b1 = torch.rand(10, 10, requires_grad=True)
    b2 = torch.rand(10, 10, requires_grad=True)

    a1 = Tensor(b1, requires_grad=True)
    a2 = Tensor(b2, requires_grad=True)

    a3 = a1 - a2
    grad = Tensor.ones((10, 10))
    a3.backward(grad)

    b_sub = b1 - b2
    b_sub.backward(torch.ones_like(b1))

    axon.testing.assert_are_close(a3, b_sub)
    axon.testing.assert_are_close(a1.grad, b1.grad)
    axon.testing.assert_are_close(a2.grad, b2.grad)


def test_broadcast_sub():
    b1 = torch.rand(10, 10, requires_grad=True)
    b2 = torch.rand(1, 10, requires_grad=True)

    a1 = Tensor(b1, requires_grad=True)  # Shape: (10, 10)
    a2 = Tensor(b2, requires_grad=True)  # Shape: (1, 10)
    a3 = a1 - a2
    grad = Tensor.ones((10, 10))
    a3.backward(grad)

    b_sub = b1 - b2
    b_sub.backward(torch.ones_like(b1))

    axon.testing.assert_are_close(a3, b_sub)
    axon.testing.assert_are_close(a1.grad, b1.grad)
    axon.testing.assert_are_close(a2.grad, b2.grad)
