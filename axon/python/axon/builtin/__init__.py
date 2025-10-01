from axon import Tensor


def mse_loss(y: Tensor, y_hat: Tensor):
    squared_err = (y - y_hat) ** 2
    return squared_err.mean()


def cross_entropy(logits: Tensor, targets_one_hot: Tensor, dim: int = 1) -> Tensor:
    probs = logits.softmax(dim=dim)
    log_probs = probs.log()
    nll = -(targets_one_hot * log_probs).sum(dim=dim)
    return nll.mean()
