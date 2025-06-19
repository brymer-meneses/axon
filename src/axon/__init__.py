from typing import List, TypeAlias
from axon._cpp import Tensor, Context

import numpy as np

TensorLike: TypeAlias = np.ndarray | List

def tensor(data: TensorLike, requires_grad: bool) -> Tensor:
    current_context = Context._get_current()
    if current_context == None:
        raise RuntimeError("Cannot invoke this function without a context.")

    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)

    return current_context._create_tensor(data, requires_grad)
    
__all__ = [
    "tensor",
    "Context",
    "Tensor",
]
