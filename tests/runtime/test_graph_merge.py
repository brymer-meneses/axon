import axon
from axon import Tensor


def test_backward_through_merged_graph():
    # Build two independent traces with distinct parameters.
    a = Tensor.randn((4, 4), requires_grad=True)
    b = Tensor.randn((4, 4), requires_grad=True)

    # Create non-root lazy tensors tied to separate sessions.
    y1 = (a * a).sum()  # session A
    y2 = (b * b).sum()  # session B

    # Merge graphs by combining lazy tensors from different sessions.
    z = y1 + y2

    # Backward through the merged graph should succeed and populate both grads.
    z.backward()

    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape

    # Optional sanity: perform a no-grad accumulate on each parameter.
    with axon.no_grad():
        a.accumulate(a.grad)
        b.accumulate(b.grad)

