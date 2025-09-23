import torch
import torch.nn.functional as F

import axon
from axon import Tensor


def test_softmax():
    b = torch.randn(100, 100, 100)
    t = Tensor(b, requires_grad=True)

    s0 = t.softmax(0)
    s1 = t.softmax(1)
    s2 = t.softmax(2)

    axon.testing.assert_are_close(s0, F.softmax(b, 0))
    axon.testing.assert_are_close(s1, F.softmax(b, 1))
    axon.testing.assert_are_close(s2, F.softmax(b, 2))
