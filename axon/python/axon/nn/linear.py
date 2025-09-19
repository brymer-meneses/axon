import math
import axon.nn as nn
from axon import Tensor


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        fan_in = math.sqrt(1 / in_features)

        self.W = Tensor.randn((in_features, out_features), requires_grad=True) * fan_in
        self.B = Tensor.zeros((out_features,), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W + self.B
