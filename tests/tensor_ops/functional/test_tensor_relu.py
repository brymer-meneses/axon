import torch
import torch.nn.functional as torch_F

import axon
import axon.nn.functional as axon_F

from axon import Tensor


def test_relu():
    t0 = torch.randn(10, 10, requires_grad=True)
    t1 = Tensor(t0, requires_grad=True)

    r0 = torch_F.relu(t0)
    r1 = axon_F.relu(t1)

    r0.backward(torch.ones_like(r0))
    r1.backward(Tensor.ones(r1.shape))

    axon.testing.assert_are_close(r1, r0)
    axon.testing.assert_are_close(t1.grad, t0.grad)
