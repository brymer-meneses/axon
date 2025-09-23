import axon
import torch

from axon import Tensor


def test_matmul():
    b1 = torch.randn(10, 5, requires_grad=True)
    b2 = torch.randn(5, 10, requires_grad=True)

    a1 = Tensor(b1, requires_grad=True)
    a2 = Tensor(b2, requires_grad=True)

    a3 = a1 @ a2
    grad = Tensor.ones((10, 10))
    a3.backward(grad)

    b_mm = b1 @ b2
    b_mm.backward(torch.ones(10, 10))

    axon.testing.assert_are_close(a1.grad, b1.grad)
    axon.testing.assert_are_close(a2.grad, b2.grad)
    axon.testing.assert_are_close(a3, b_mm)


def test_matmul_broadcast():
    b1 = torch.randn(3, 10, 20, requires_grad=True)
    b2 = torch.randn(20, 10, requires_grad=True)
    a1 = Tensor(b1, requires_grad=True)
    a2 = Tensor(b2, requires_grad=True)

    l = a1 @ a2
    grad = Tensor.ones((3, 10, 10))
    l.backward(grad)

    b_mm = b1 @ b2
    b_mm.backward(torch.ones(3, 10, 10))

    axon.testing.assert_are_close(a1.grad, b1.grad)
    axon.testing.assert_are_close(a2.grad, b2.grad)
    axon.testing.assert_are_close(l, b_mm)


def test_matmul_broadcast_different_batch():
    b1 = torch.randn(1, 15, 10, requires_grad=True)
    b2 = torch.randn(4, 10, 15, requires_grad=True)
    a1 = Tensor(b1, requires_grad=True)
    a2 = Tensor(b2, requires_grad=True)
    grad = Tensor.ones((4, 15, 15))

    result = a1 @ a2
    result.backward(grad)

    b_mm = b1 @ b2
    b_mm.backward(torch.ones(4, 15, 15))

    axon.testing.assert_are_close(a1.grad, b1.grad)
    axon.testing.assert_are_close(a2.grad, b2.grad)
    axon.testing.assert_are_close(result, b_mm)
