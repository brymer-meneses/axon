import torch
import torch.nn.functional as torch_F

import axon
import axon.nn.functional as axon_F

from axon import Tensor


def test_cross_entropy_forward_and_backward():
    torch.manual_seed(0)

    N, C = 8, 5
    base = torch.randn(N, C)
    logits_t = base.clone().detach().requires_grad_(True)
    targets_idx = torch.randint(0, C, (N,))
    one_hot = torch_F.one_hot(targets_idx, num_classes=C).to(logits_t.dtype)

    logits = Tensor(base, requires_grad=True)
    targets = Tensor(one_hot)

    loss_axon = axon_F.cross_entropy(logits * 0.5, targets)
    loss_torch = torch_F.cross_entropy(logits_t * 0.5, targets_idx)

    loss_torch.backward()
    loss_axon.backward()

    axon.testing.assert_are_close(loss_axon, loss_torch, 1e-4)
    axon.testing.assert_are_close(logits.grad, logits_t.grad, 5e-3)
