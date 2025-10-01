import torch
import torch.nn.functional as F

from axon import Tensor
import axon


def test_cross_entropy_forward_and_backward():
    torch.manual_seed(0)

    N, C = 8, 5
    base = torch.randn(N, C)
    logits_t = base.clone().detach().requires_grad_(True)
    targets_idx = torch.randint(0, C, (N,))
    one_hot = F.one_hot(targets_idx, num_classes=C).to(logits_t.dtype)

    logits = Tensor(base, requires_grad=True)
    targets = Tensor(one_hot)

    # Compute losses
    loss_axon = axon.builtin.cross_entropy(logits * 0.5, targets)
    loss_torch = F.cross_entropy(logits_t * 0.5, targets_idx)

    # Backward
    loss_torch.backward()
    loss_axon.backward()

    # Compare values and gradients
    axon.testing.assert_are_close(loss_axon, loss_torch, 1e-4)
    axon.testing.assert_are_close(logits.grad, logits_t.grad, 5e-3)
