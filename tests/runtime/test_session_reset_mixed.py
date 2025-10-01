import pytest
from axon import Tensor


def test_matmul_mixed_root_nonroot_after_backward_repro():
    # Build an initial trace: y1 = x @ a; loss = mean(y1); backward()
    x = Tensor.randn((5, 2))  # root, no grads
    a = Tensor.randn((2, 3), requires_grad=True)
    y1 = x @ a  # y1 is non-root (lazy), tied to a trace session
    loss = y1.mean()
    loss.backward()

    # Start a new trace input and attempt to mix with old non-root y1
    z = Tensor.randn((2, 5))
    _ = z @ y1

    assert a.grad is not None
