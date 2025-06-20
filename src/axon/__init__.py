from typing import List, TypeAlias

import axon._cpp
import numpy as np

from axon._cpp import Tensor

TensorLike: TypeAlias = np.ndarray | List

def tensor(data: TensorLike, requires_grad: bool = False) -> Tensor:
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    #
    return axon._cpp._create_tensor(data, requires_grad)
    
__all__ = [
    "tensor",
    "Tensor",
]
