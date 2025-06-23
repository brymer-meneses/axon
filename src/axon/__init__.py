import typing
import numpy as np

import axon._cpp as cpp

TensorConvertable: typing.TypeAlias = np.ndarray | typing.List

def tensor(data: TensorConvertable, requires_grad=False) -> cpp.Tensor:
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    return cpp.create_tensor(data, requires_grad)
