from axon import Tensor


def mse_loss(y: Tensor, y_hat: Tensor):
    squared_err = (y - y_hat) ** 2
    return squared_err.mean()
