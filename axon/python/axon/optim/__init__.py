import axon

from axon import Tensor
from abc import ABC, abstractmethod
from typing import List


class Optim(ABC):
    def __init__(self, parameters: List[Tensor]) -> None:
        self._parameters = parameters

    def parameters(self) -> List[Tensor]:
        return self._parameters

    @abstractmethod
    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        for param in self._parameters:
            if param.requires_grad:
                param.zero_grad()


class SGD(Optim):
    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 1e-5,
        *,
        gamma: float = 1.0,
        decay_every: int = 1,
        min_lr: float = 0.0,
    ) -> None:
        """
        Stochastic Gradient Descent with optional step-wise LR decay.

        Args:
            parameters: model parameters to optimize
            lr: initial learning rate
            gamma: multiplicative decay factor (e.g., 0.99). 1.0 disables decay
            decay_every: apply decay every N calls to step()
            min_lr: lower bound for the learning rate
        """
        super().__init__(parameters)
        self._lr = lr
        self._gamma = gamma
        self._decay_every = max(1, int(decay_every))
        self._min_lr = min_lr
        self._step_count = 0

    @property
    def lr(self) -> float:
        return self._lr

    def step(self) -> None:
        with axon.no_grad():
            for param in self.parameters():
                lr = Tensor.fill_like(self._lr, param.shape)
                param.accumulate(-param.grad * lr)

        # Update LR after applying gradients
        self._step_count += 1
        if self._gamma < 1.0 and (self._step_count % self._decay_every) == 0:
            self._lr = max(self._lr * self._gamma, self._min_lr)
