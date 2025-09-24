import axon

from axon import Tensor


def test_loop_caching():
    a = Tensor([1, 2, 3], requires_grad=False)
    b = Tensor([1, 2, 3], requires_grad=False)

    compiled_functions = axon.runtime.total_number_of_compiled_functions()

    for _ in range(100):
        c = a + b
        c.evaluate()

    assert axon.runtime.total_number_of_compiled_functions() == compiled_functions + 1

    a = Tensor([1, 2, 3, 4], requires_grad=False)
    b = Tensor([1, 2, 3, 5], requires_grad=False)

    for _ in range(100):
        c = a * b
        c.evaluate()

    assert axon.runtime.total_number_of_compiled_functions() == compiled_functions + 2
