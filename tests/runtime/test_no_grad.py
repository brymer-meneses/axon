import axon
from axon import Tensor


def test_no_grad_resource_manager():
    a = Tensor.randn((10, 10), requires_grad=True)
    b = Tensor.randn((10, 10), requires_grad=True)

    with axon.no_grad():
        c = a + b
        assert not c.requires_grad

    c = a + b
    assert c.requires_grad


def test_no_grad_decorator():
    @axon.no_grad()
    def func(a, b):
        return a + b

    def func1(a, b):
        return a + b

    a = Tensor.randn((10, 10), requires_grad=True)
    b = Tensor.randn((10, 10), requires_grad=True)

    c = func(a, b)
    d = func1(a, b)

    assert not c.requires_grad
    assert d.requires_grad
