import numpy as np
import pytest
import torch

import axon
from axon import Tensor, dtype

import axon.nn.functional as F


@pytest.mark.parametrize("torch_dtype", [torch.int32, torch.int64])
def test_integer_addition(torch_dtype):
    lhs = torch.randint(-10, 10, (4, 5), dtype=torch_dtype)
    rhs = torch.randint(-10, 10, (4, 5), dtype=torch_dtype)

    ax_lhs = Tensor(lhs)
    ax_rhs = Tensor(rhs)

    result = ax_lhs + ax_rhs
    expected = (lhs + rhs).numpy()

    axon.testing.assert_are_close(result, expected)


@pytest.mark.parametrize("torch_dtype", [torch.int32, torch.int64])
def test_integer_elementwise_sub_and_mul(torch_dtype):
    lhs = torch.randint(-10, 10, (3, 4), dtype=torch_dtype)
    rhs = torch.randint(-10, 10, (3, 4), dtype=torch_dtype)

    ax_lhs = Tensor(lhs)
    ax_rhs = Tensor(rhs)

    sub_result = ax_lhs - ax_rhs
    mul_result = ax_lhs * ax_rhs

    axon.testing.assert_are_close(sub_result, (lhs - rhs).numpy())
    axon.testing.assert_are_close(mul_result, (lhs * rhs).numpy())


@pytest.mark.parametrize(
    "torch_dtype,scalar",
    [
        (torch.int32, 3),
        (torch.int64, 7),
    ],
)
def test_integer_scalar_mul(torch_dtype, scalar):
    base = torch.randint(-5, 5, (2, 3), dtype=torch_dtype)
    tensor = Tensor(base)

    result = tensor * scalar
    expected = (base * scalar).numpy()

    axon.testing.assert_are_close(result, expected)


def test_integer_scalar_mul_rejects_float_scalar():
    tensor = Tensor(torch.arange(6, dtype=torch.int32).reshape(2, 3))

    with pytest.raises(ValueError):
        tensor * 1.5


@pytest.mark.parametrize("torch_dtype", [torch.int32, torch.int64])
def test_integer_sum(torch_dtype):
    data = torch.randint(-10, 10, (6,), dtype=torch_dtype)
    tensor = Tensor(data)

    summed = tensor.sum()
    expected = np.array(data.sum().item(), dtype=data.numpy().dtype)

    axon.testing.assert_are_close(summed, expected)


@pytest.mark.parametrize("torch_dtype", [torch.int32, torch.int64])
def test_float_only_ops_raise(torch_dtype):
    data = torch.randint(-3, 3, (3, 3), dtype=torch_dtype)
    tensor = Tensor(data)

    with pytest.raises(RuntimeError):
        tensor.log()
    with pytest.raises(RuntimeError):
        F.softmax(tensor, 0)
    with pytest.raises(RuntimeError):
        F.relu(tensor)
    with pytest.raises(RuntimeError):
        tensor.mean()
    with pytest.raises(RuntimeError):
        tensor.pow(2.0)


def test_randn_disallows_integer_dtype():
    with pytest.raises(ValueError):
        Tensor.randn((2, 2), dtype=dtype.int32)


def test_boolean_tensor_construction():
    ones = Tensor.ones((3,), dtype=dtype.bool)
    axon.testing.assert_are_close(ones, np.ones((3,), dtype=np.bool))

    zeros = Tensor.zeros((2,), dtype=dtype.bool)
    axon.testing.assert_are_close(zeros, np.zeros((2,), dtype=np.bool))
