import axon
import torch

from axon import Tensor


def test_add():
    b1 = torch.randn(10, 10)
    b2 = torch.randn(10, 10)

    a1 = Tensor(b1, requires_grad=True)
    a2 = Tensor(b2, requires_grad=True)

    a3 = a1 + a2
    a3.backward(Tensor.ones((10, 10)))

    axon.testing.assert_are_close(a3, b1 + b2)
    axon.testing.assert_are_close(a1.grad, torch.ones(a3.shape))
    axon.testing.assert_are_close(a2.grad, torch.ones(a3.shape))


def test_broadcast_add():
    b1 = torch.randn(10, 10)
    b2 = torch.randn(1, 10)

    a1 = Tensor(b1, requires_grad=True)  # Shape: (10, 10)
    a2 = Tensor(b2, requires_grad=True)  # Shape: (1, 10)
    a3 = a1 + a2
    a3.backward(Tensor.ones((10, 10)))

    axon.testing.assert_are_close(a3, b1 + b2)
    axon.testing.assert_are_close(a1.grad, torch.ones(a3.shape))

    expected_a2_grad = torch.sum(torch.ones_like(b1), axis=0, keepdims=True)
    axon.testing.assert_are_close(a2.grad, expected_a2_grad)
