import enum
from typing import Annotated, overload

from numpy.typing import ArrayLike


class dtype(enum.Enum):
    float32 = 0

    float64 = 1

    bool = 2

    int32 = 3

    int64 = 4

float32: dtype = dtype.float32

float64: dtype = dtype.float64

bool: dtype = dtype.bool

int32: dtype = dtype.int32

int64: dtype = dtype.int64

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

    def __neg__(self) -> Tensor: ...

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

    @overload
    def __mul__(self, arg: float, /) -> Tensor: ...

    @overload
    def __mul__(self, arg: int, /) -> Tensor: ...

    @overload
    def __mul__(self, arg: int, /) -> Tensor: ...

    def __sub__(self, arg: Tensor, /) -> Tensor: ...

    @overload
    def __rmul__(self, arg: float, /) -> Tensor: ...

    @overload
    def __rmul__(self, arg: float, /) -> Tensor: ...

    @overload
    def __rmul__(self, arg: int, /) -> Tensor: ...

    @overload
    def __rmul__(self, arg: int, /) -> Tensor: ...

    @overload
    def pow(self, exponent: float) -> Tensor: ...

    @overload
    def pow(self, exponent: float) -> Tensor: ...

    @overload
    def __pow__(self, arg: float, /) -> Tensor: ...

    @overload
    def __pow__(self, arg: float, /) -> Tensor: ...

    def __matmul__(self, arg: Tensor, /) -> Tensor: ...

    def softmax(self, dim: int) -> Tensor: ...

    def relu(self) -> Tensor: ...

    def log(self) -> Tensor: ...

    def accumulate(self, value: Tensor) -> None: ...

    def sum(self, dim: int | None = None, keepdims: bool = False) -> Tensor: ...

    def mean(self, dim: int | None = None, keepdims: bool = False) -> Tensor: ...

    def item(self) -> object: ...

    @staticmethod
    def fill_like(value: object, shape: tuple, requires_grad: bool = False, dtype: dtype = dtype.float32) -> Tensor: ...

    @staticmethod
    def ones(shape: tuple, requires_grad: bool = False, dtype: dtype = dtype.float32) -> Tensor: ...

    @staticmethod
    def zeros(shape: tuple, requires_grad: bool = False, dtype: dtype = dtype.float32) -> Tensor: ...

    @staticmethod
    def randn(shape: tuple, requires_grad: bool = False, dtype: dtype = dtype.float32) -> Tensor: ...

def assert_are_close(tensor: Tensor, ndarray: Annotated[ArrayLike, dict(order='C', writable=False)], tolerance: float = 1e-05) -> None: ...

def total_number_of_compiled_functions() -> int: ...

def set_emit_grad(arg: bool, /) -> None: ...
