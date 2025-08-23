from . import axon_bindings as bindings

import typing
import numpy as np

def tensor(ndarray, requires_grad=False) -> bindings.Tensor:
    if not isinstance(ndarray, np.ndarray):
        ndarray = np.array(ndarray)
    return bindings.create_tensor(ndarray, requires_grad)

def rand(shape, requires_grad=False) -> bindings.Tensor:
    return bindings.create_rand(shape, requires_grad)

def compile(func: typing.Callable) -> typing.Callable:
    return func
