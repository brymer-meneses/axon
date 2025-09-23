import torch
import torch.nn.functional as F

import axon
from axon import Tensor


def test_softmax_forward_backward_all_axes():
    # Test forward and backward across all axes, reusing the torch computation
    # for both assertion and gradient.
    shape = (20, 10, 5)
    for axis in (0, 1, 2):
        b = torch.randn(*shape, requires_grad=True)
        t = Tensor(b, requires_grad=True)

        ax = t.softmax(axis)
        torch_soft = F.softmax(b, axis)

        # Forward equality
        axon.testing.assert_are_close(ax, torch_soft)

        # Backward using the same torch_soft value
        ax.backward(Tensor.ones(shape))
        torch_soft.backward(torch.ones_like(b))

        axon.testing.assert_are_close(t.grad, b.grad)
