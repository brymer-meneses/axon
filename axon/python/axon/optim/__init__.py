import axon

from axon import Tensor
from abc import ABC, abstractmethod
from typing import List


class Optim(ABC):
    def __init__(self, parameters: List[Tensor]) -> None:
        self._parameters = parameters

    @abstractmethod
    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        for param in self._parameters:
            param.zero_grad()


class SGD(Optim):
    def __init__(self, parameters: List[Tensor], lr=1e-5) -> None:
        super().__init__(parameters)
        self._lr = lr

    def step(self) -> None:
        with axon.no_grad():
            for param in self._parameters:
                param.accumulate(param.grad * self._lr)
