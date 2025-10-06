import torch
import torch.nn.functional as torch_F

import axon
import axon.nn.functional as axon_F

from axon import Tensor


def test_mse_loss():
    t0 = torch.randn(100, 100, 100, requires_grad=True)
    t1 = torch.randn(100, 100, 100, requires_grad=True)

    a0 = Tensor(t0, requires_grad=True)
    a1 = Tensor(t1, requires_grad=True)

    t2 = torch_F.mse_loss(t0, t1)
    a2 = axon_F.mse_loss(a0, a1)

    t2.backward()
    a2.backward()

    axon.testing.assert_are_close(a2, t2)
    axon.testing.assert_are_close(a0.grad, t0.grad)
    axon.testing.assert_are_close(a1.grad, t1.grad)
