import torch
import axon
import numpy as np
import pytest

from axon import Tensor


def test_tensor_accumulate():
    sink_ = torch.randn(10, 10)
    source_ = torch.randn(10, 10)

    sink = Tensor(sink_)
    source = Tensor(source_)

    with axon.no_grad():
        sink.accumulate(source)

    axon.testing.assert_are_close(sink, sink_ + source_)


def test_tensor_accumulate_with_nonroot_source():
    sink_arr = np.random.randn(6, 6).astype(np.float32)
    base_arr = np.random.randn(6, 6).astype(np.float32)

    sink = Tensor(sink_arr, requires_grad=True)
    base = Tensor(base_arr)
    source = base * 2.0

    with axon.no_grad():
        sink.accumulate(source)

    expected = sink_arr + base_arr * 2.0
    axon.testing.assert_are_close(sink, expected)


def test_tensor_accumulate_raises_on_lazy_sink():
    base = Tensor.randn((4, 4), requires_grad=True)
    lazy_sink = base * 1.0

    source = Tensor.randn((4, 4))

    with axon.no_grad():
        with pytest.raises(Exception):
            lazy_sink.accumulate(source)


def test_accumulate_non_parameter_sink_triggers_codegen_assert():
    a = Tensor.randn((3, 3), requires_grad=True)
    sink = a * 1.0
    sink.evaluate()

    source = sink * 2.0

    with axon.no_grad():
        sink.accumulate(source)
