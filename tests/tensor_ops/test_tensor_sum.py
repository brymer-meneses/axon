from axon import Tensor

import torch
import axon


def test_sum_keepdims():
    b = torch.randn(10, 10, requires_grad=True)
    t = Tensor(b, requires_grad=True)

    result = t.sum(0, keepdims=True)
    result.backward(Tensor.ones((1, 10)))

    b_sum_keepdims = b.sum(0, keepdims=True)
    b_sum_keepdims.backward(torch.ones(1, 10))

    axon.testing.assert_are_close(result, b_sum_keepdims)
    axon.testing.assert_are_close(t.grad, b.grad)


def test_sum():
    b = torch.randn(10, 10, requires_grad=True)
    t = Tensor(b, requires_grad=True)

    result = t.sum(0)
    result.backward(Tensor.ones((10,)))

    b_sum = b.sum(0)
    b_sum.backward(torch.ones(10))

    axon.testing.assert_are_close(result, b_sum)
    axon.testing.assert_are_close(t.grad, b.grad)


def test_sum_reduce_all():
    t0 = torch.randn((100, 100, 100), requires_grad=True)
    a0 = Tensor(t0, requires_grad=True)

    t1 = t0.sum()
    a1 = a0.sum()

    t1.backward()
    a1.backward()

    axon.testing.assert_are_close(a1, t1, 1e-3)
    axon.testing.assert_are_close(a0.grad, t0.grad)
