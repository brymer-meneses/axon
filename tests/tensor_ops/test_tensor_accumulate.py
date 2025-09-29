import torch
import axon
from axon import Tensor


def test_tensor_accumulate():
    sink_ = torch.randn(10, 10)
    source_ = torch.randn(10, 10)

    sink = Tensor(sink_)
    source = Tensor(source_)

    with axon.no_grad():
        sink.accumulate(source)

    axon.testing.assert_are_close(sink, sink_ + source_)
