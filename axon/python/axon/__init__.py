from . import axon_bindings as bindings

import numpy as np

def tensor(ndarray, requires_grad=False) -> bindings.Tensor:
    if not isinstance(ndarray, np.ndarray):
        ndarray = np.array(ndarray)
    return bindings.create_tensor(ndarray, requires_grad)



