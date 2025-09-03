import atexit


from .axon_bindings import LoweringLevel
from .tensor import tensor
from .jit import jit, LoweringOps

__all__ = [
    "jit",
    "tensor"
]

def _cleanup() -> None:
    axon_bindings._clear_current_graph()
    
atexit.register(_cleanup)
