import torch

import axon
from axon import Tensor


def test_pow_forward_backward():
    b = torch.randn(10, 10, requires_grad=True)
    t = Tensor(b, requires_grad=True)

    exp = 3.0

    r_t = t.pow(exp)
    r_b = b.pow(exp)

    r_t.backward(Tensor.ones(r_t.shape))
    r_b.backward(torch.ones_like(b))

    axon.testing.assert_are_close(r_t, r_b)
    axon.testing.assert_are_close(t.grad, b.grad)


def test_pow_operator():
    b = torch.randn(5, 7, requires_grad=True)
    t = Tensor(b, requires_grad=True)

    exp = 2.0

    r_t = t ** exp
    r_b = b ** exp

    r_t.backward(Tensor.ones(r_t.shape))
    r_b.backward(torch.ones_like(b))

    axon.testing.assert_are_close(r_t, r_b)
    axon.testing.assert_are_close(t.grad, b.grad)

