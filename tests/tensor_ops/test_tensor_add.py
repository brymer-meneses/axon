import axon
import torch

from axon import Tensor


def test_add():
    b1 = torch.randn(10, 10, requires_grad=True)
    b2 = torch.randn(10, 10, requires_grad=True)

    a1 = Tensor(b1, requires_grad=True)
    a2 = Tensor(b2, requires_grad=True)

    a3 = a1 + a2
    a3.backward(Tensor.ones((10, 10)))

    b_add = b1 + b2
    b_add.backward(torch.ones_like(b1))

    axon.testing.assert_are_close(a3, b_add)
    axon.testing.assert_are_close(a1.grad, b1.grad)
    axon.testing.assert_are_close(a2.grad, b2.grad)


def test_broadcast_add():
    b1 = torch.randn(10, 10, requires_grad=True)
    b2 = torch.randn(1, 10, requires_grad=True)

    a1 = Tensor(b1, requires_grad=True)  # Shape: (10, 10)
    a2 = Tensor(b2, requires_grad=True)  # Shape: (1, 10)
    a3 = a1 + a2
    a3.backward(Tensor.ones((10, 10)))

    b_add = b1 + b2
    b_add.backward(torch.ones_like(b1))

    axon.testing.assert_are_close(a3, b_add)
    axon.testing.assert_are_close(a1.grad, b1.grad)
    axon.testing.assert_are_close(a2.grad, b2.grad)


def test_broadcast_add_both_sides():
    # (1, 4) + (4, 1) -> (4, 4)
    b1 = torch.randn(1, 4, requires_grad=True)
    b2 = torch.randn(4, 1, requires_grad=True)

    a1 = Tensor(b1, requires_grad=True)
    a2 = Tensor(b2, requires_grad=True)

    a3 = a1 + a2
    a3.backward(Tensor.ones(a3.shape))

    b_add = b1 + b2
    b_add.backward(torch.ones_like(b_add))

    axon.testing.assert_are_close(a3, b_add)
    axon.testing.assert_are_close(a1.grad, b1.grad)
    axon.testing.assert_are_close(a2.grad, b2.grad)


def test_add_incompatible_shapes():
    b1 = torch.randn(2, 3, requires_grad=True)
    b2 = torch.randn(2, 2, requires_grad=True)
    a1 = Tensor(b1, requires_grad=True)
    a2 = Tensor(b2, requires_grad=True)

    import pytest
    with pytest.raises(Exception):
        _ = a1 + a2
