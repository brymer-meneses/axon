import enum
from typing import Annotated, overload

from numpy.typing import ArrayLike


class dtype(enum.Enum):
    float32 = 0

    float64 = 1

float32: dtype = dtype.float32

float64: dtype = dtype.float64

class LoweringLevel(enum.Enum):
    Axon = 0

    Standard = 1

    Linalg = 2

    Loops = 3

    LLVM = 4

Axon: LoweringLevel = LoweringLevel.Axon

Standard: LoweringLevel = LoweringLevel.Standard

Linalg: LoweringLevel = LoweringLevel.Linalg

Loops: LoweringLevel = LoweringLevel.Loops

LLVM: LoweringLevel = LoweringLevel.LLVM

def inspect_ir(tensor: Tensor, level: LoweringLevel) -> None: ...

class Tensor:
    def __init__(self, data: object, requires_grad: bool = False, dtype: dtype = dtype.float32) -> None: ...

    @property
    def shape(self) -> tuple: ...

    @property
    def requires_grad(self) -> bool: ...

    @property
    def grad(self) -> Tensor: ...

    @property
    def is_evaluated(self) -> bool: ...

    def zero_grad(self) -> None: ...

    def evaluate(self) -> None: ...

    def backward(self, grad: Tensor | None = None) -> None: ...

    def __repr__(self) -> str: ...

    def __eq__(self, arg: Tensor, /) -> Tensor: ...

    def __ne__(self, arg: Tensor, /) -> Tensor: ...

    def __le__(self, arg: Tensor, /) -> Tensor: ...

    def __lt__(self, arg: Tensor, /) -> Tensor: ...

    def __ge__(self, arg: Tensor, /) -> Tensor: ...

    def __gt__(self, arg: Tensor, /) -> Tensor: ...

    def __add__(self, arg: Tensor, /) -> Tensor: ...

    @overload
    def __mul__(self, arg: Tensor, /) -> Tensor: ...

    @overload
    def __mul__(self, arg: float, /) -> Tensor: ...

    def __sub__(self, arg: Tensor, /) -> Tensor: ...

    def __rmul__(self, arg: float, /) -> Tensor: ...

    def __matmul__(self, arg: Tensor, /) -> Tensor: ...

    def softmax(self, dim: int) -> Tensor: ...

    def sum(self, dim: int, keepdims: bool = False) -> Tensor: ...

    def relu(self) -> Tensor: ...

    @staticmethod
    def ones(shape: tuple, requires_grad: bool = False, dtype: dtype = dtype.float32) -> Tensor: ...

    @staticmethod
    def zeros(shape: tuple, requires_grad: bool = False, dtype: dtype = dtype.float32) -> Tensor: ...

    @staticmethod
    def randn(shape: tuple, requires_grad: bool = False, dtype: dtype = dtype.float32) -> Tensor: ...

def assert_are_close(tensor: Tensor, ndarray: Annotated[ArrayLike, dict(order='C', writable=False)], tolerance: float = 1e-05) -> None: ...

def total_number_of_compiled_functions() -> int: ...
