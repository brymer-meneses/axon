import axon
import torch

from axon import Tensor


def test_scalar_mul():
    x = torch.rand(5, 5, requires_grad=True)
    t = Tensor(x, requires_grad=True)

    result = t * 5.0
    result.backward(Tensor.ones((5, 5)))

    x_scaled = x * 5
    x_scaled.backward(torch.ones_like(x))

    axon.testing.assert_are_close(result, x_scaled)
    axon.testing.assert_are_close(t.grad, x.grad)


def test_scalar_right_mul():
    x = torch.rand(5, 5, requires_grad=True)
    t = Tensor(x, requires_grad=True)

    result = 5.0 * t
    result.backward(Tensor.ones((5, 5)))

    x_scaled = x * 5
    x_scaled.backward(torch.ones_like(x))

    axon.testing.assert_are_close(result, x_scaled)
    axon.testing.assert_are_close(t.grad, x.grad)
