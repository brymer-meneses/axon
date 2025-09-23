import torch

import axon
from axon import Tensor


def _assert_mask_close(ax_mask, torch_mask):
    # Axon compare returns a numeric mask; convert torch bool to float
    axon.testing.assert_are_close(ax_mask, torch_mask.to(dtype=torch.float32))


def test_compare_all_predicates():
    a = torch.randn(8, 16)
    b = a.clone()
    # Create a mix of equal and non-equal positions
    flip = torch.rand_like(a) > 0.5
    b[flip] = b[flip] + 1.0

    ta = Tensor(a)
    tb = Tensor(b)

    # <, <=, >, >=
    _assert_mask_close(ta < tb, a < b)
    _assert_mask_close(ta <= tb, a <= b)
    _assert_mask_close(ta > tb, a > b)
    _assert_mask_close(ta >= tb, a >= b)

    # ==, !=
    _assert_mask_close(ta == tb, a == b)
    _assert_mask_close(ta != tb, a != b)

