import torch
from axon import Tensor
import axon


def test_log_forward_and_backward_matches_torch():
    torch.manual_seed(0)

    # Ensure strictly positive inputs to avoid domain issues.
    # Keep a leaf tensor `r` for PyTorch so gradients are populated on it.
    r = torch.rand(6, 7, requires_grad=True)
    t = r + 0.1

    a = Tensor(t, requires_grad=True)

    # Forward
    y_t = torch.log(t)
    y_a = a.log()

    axon.testing.assert_are_close(y_a, y_t)

    # Backward: sum to get a scalar loss; dl/dx = 1/x
    (y_t.sum()).backward()
    (y_a.sum()).backward()

    # Compare against grad on the leaf `r` (equal to grad w.r.t. `t`).
    axon.testing.assert_are_close(a.grad, r.grad)
