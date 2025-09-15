import enum


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

class Tensor:
    def __init__(self, data: object, requires_grad: bool = False, dtype: dtype = dtype.float64) -> None: ...

    @property
    def shape(self) -> tuple: ...

    @property
    def requires_grad(self) -> bool: ...

    @property
    def is_evaluated(self) -> bool: ...

    def backward(self, arg: Tensor, /) -> None: ...

    def __repr__(self) -> str: ...

    def __add__(self, arg: Tensor, /) -> Tensor: ...

    def __mul__(self, arg: Tensor, /) -> Tensor: ...

    def __sub__(self, arg: Tensor, /) -> Tensor: ...

    @staticmethod
    def ones(shape: tuple, requires_grad: bool = False, dtype: dtype = dtype.float64) -> Tensor: ...

    @staticmethod
    def zeros(shape: tuple, requires_grad: bool = False, dtype: dtype = dtype.float64) -> Tensor: ...

    @staticmethod
    def randn(shape: tuple, requires_grad: bool = False, dtype: dtype = dtype.float64) -> Tensor: ...
