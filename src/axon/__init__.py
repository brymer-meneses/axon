from . import _axon_cpp


class Tensor:

    def __init__(self, ctx, requires_grad) -> None:
        self._requires_grad = requires_grad
        self._ctx = ctx

    def __add__(self, other: "Tensor") -> "Tensor":
        return self._ctx.add(self, other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        return self._ctx.mul(self, other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self._ctx.matmul(self, other)

