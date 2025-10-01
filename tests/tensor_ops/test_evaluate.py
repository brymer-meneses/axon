from axon import Tensor
import pytest


def test_evaluate_materialized_tensor():
    a = Tensor.randn((10,))

    with pytest.raises(Exception):
        a.evaluate()
