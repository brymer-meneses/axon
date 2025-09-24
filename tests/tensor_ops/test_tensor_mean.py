import axon
import torch

from axon import Tensor


def test_mean_in_dim():
    t0 = torch.randn((100, 100), requires_grad=True)
    a0 = Tensor(t0, requires_grad=True)

    t1 = t0.mean(dim=0)
    a1 = a0.mean(dim=0)

    t1.backward(torch.ones_like(t1))
    a1.backward(Tensor.ones(a1.shape))

    axon.testing.assert_are_close(a1, t1)
    axon.testing.assert_are_close(a0.grad, t0.grad)


def test_mean_reduce_all():
    t0 = torch.randn((100, 100, 100), requires_grad=True)
    a0 = Tensor(t0, requires_grad=True)

    t1 = t0.mean()
    a1 = a0.mean()

    t1.backward()
    a1.backward()

    axon.testing.assert_are_close(a1, t1)
    axon.testing.assert_are_close(a0.grad, t0.grad)
