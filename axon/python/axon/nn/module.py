from typing import Any, List

from axon._core import Tensor


class Module:
    def __init__(self) -> None:
        self._parameters: List[Tensor] = []

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Tensor):
            self._parameters.append(value)
        elif isinstance(value, Module):
            self._parameters.extend(value._parameters)

        self.__dict__[name] = value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    @property
    def parameters(self) -> List[Tensor]:
        return self._parameters

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Inherting from `axon.Module` requires the forward method to be implemented."
        )
