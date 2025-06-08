import axon
import pytest

def test_invalid_function():
    with pytest.raises(ValueError):
        axon.func(1)

def test_valid_function():
    axon.func(lambda x: x)
