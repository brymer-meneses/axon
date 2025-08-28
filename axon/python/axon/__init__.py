import atexit

from . import axon_bindings as bindings

from .axon_bindings import LoweringOps
from .tensor import tensor
from .jit import jit

__all__ = [
    "jit",
    "tensor"
]

def _cleanup() -> None:
    bindings._clear_current_graph()
    
atexit.register(_cleanup)
