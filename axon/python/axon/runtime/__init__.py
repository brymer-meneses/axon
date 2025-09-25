from typing import Callable
import axon._core as _core


def total_number_of_compiled_functions() -> int:
    return _core.total_number_of_compiled_functions()


class no_grad:
    def __enter__(self) -> None:
        _core.set_emit_grad(False)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _core.set_emit_grad(True)

    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper
