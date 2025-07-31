import typing 
import _cpp 

from axon.tensor import Tensor, Parameter

class Module:

    def __init__(self) -> None:
        self._module = _cpp.Module()

    def __setattr__(self, name: str, value: typing.Any) -> None:
        if isinstance(value, Parameter):
            self._module.declare_parameter(value)

    def __call__(self, *args, **kwargs) -> typing.Any:
        return self.forward(args, kwargs)

    def forward(*args, **kwargs) -> typing.Any:
        raise NotImplementedError("Forward must be implemented for a class that inherits from `Module`")

