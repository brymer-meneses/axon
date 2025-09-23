import axon
import torch

from axon import Tensor


def test_scalar_mul():
    x = torch.rand(5, 5)
    t = Tensor(x, requires_grad=True)

    result = t * 5.0
    result.backward(Tensor.ones((5, 5)))

    axon.testing.assert_are_close(result, (x * 5).to(torch.float32))
    axon.testing.assert_are_close(t.grad, torch.ones_like(x) * 5)


def test_scalar_right_mul():
    x = torch.rand(5, 5)
    t = Tensor(x, requires_grad=True)

    result = 5.0 * t
    result.backward(Tensor.ones((5, 5)))

    axon.testing.assert_are_close(result, (x * 5).to(torch.float32))
    axon.testing.assert_are_close(t.grad, torch.ones_like(x) * 5)
