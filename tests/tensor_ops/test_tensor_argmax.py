import torch

from axon import Tensor

import axon


def test_tensor_argmax_matches_torch():
    torch.manual_seed(0)
    data = torch.randn(3, 4, 5)
    tensor = Tensor(data)

    for dim in range(data.ndim):
        axon_result = tensor.argmax(dim)
        torch_result = data.argmax(dim=dim)
        axon.testing.assert_are_close(axon_result, torch_result)


def test_tensor_argmax_keepdims():
    data = torch.randn(2, 5, 7)
    tensor = Tensor(data)

    axon_result = tensor.argmax(1, keepdims=True)
    torch_result = data.argmax(dim=1, keepdim=True)

    assert axon_result.shape == torch_result.shape
    axon.testing.assert_are_close(axon_result, torch_result)


def test_tensor_argmax_negative_axis():
    data = torch.randn(4, 3, 6)
    tensor = Tensor(data)

    axon_result = tensor.argmax(-1)
    torch_result = data.argmax(dim=-1)

    axon.testing.assert_are_close(axon_result, torch_result)
