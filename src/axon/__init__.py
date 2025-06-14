from . import _axon_cpp

class Tensor:

    def __init__(self, data, requires_grad=False) -> None:
        import inspect

        self._requires_grad = requires_grad
        self._data = data
        self._ctx = _axon_cpp.Context(inspect.stack()[1].function);


    def __add__(self, other: "Tensor") -> "Tensor":
        raise NotImplementedError

    def __mul__(self, other: "Tensor") -> "Tensor":
        raise NotImplementedError

    def __matmul__(self, other: "Tensor") -> "Tensor":
        raise NotImplementedError

