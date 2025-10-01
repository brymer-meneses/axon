import axon
from axon import Tensor


def test_scalar_tensor_repr_without_grad():
    t = Tensor.zeros((2, 2))
    s = t.sum()  # scalar
    r = repr(s)
    assert r == "tensor(0.0000)"


def test_scalar_tensor_repr_with_grad_flag():
    t = Tensor.ones((2, 2), requires_grad=True)
    s = t.sum()  # scalar, requires grad
    r = repr(s)
    assert r == "tensor(4.0000, requires_grad=True)"

