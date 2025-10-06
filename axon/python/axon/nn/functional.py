"""Neural network functional operators.

This module mimics the structure of ``torch.nn.functional`` by providing
stateless helper functions that operate on :class:`axon._core.Tensor`.
"""

from .._core import Tensor

from .. import _core

__all__ = ["mse_loss", "cross_entropy"]


def mse_loss(y: Tensor, y_hat: Tensor) -> Tensor:
    """Compute the mean squared error between ``y`` and ``y_hat``."""

    squared_err = (y - y_hat) ** 2
    return squared_err.mean()


def cross_entropy(logits: Tensor, targets_one_hot: Tensor, dim: int = 1) -> Tensor:
    """Compute the cross entropy loss from logits and one-hot targets."""

    probs = _core.softmax(logits, dim=dim)

    log_probs = probs.log()
    nll = -(targets_one_hot * log_probs).sum(dim=dim)
    return nll.mean()


def softmax(input: Tensor, dim: int) -> Tensor:
    return _core.softmax(input, dim)


def relu(input: Tensor) -> Tensor:
    return _core.relu(input)
