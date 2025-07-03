# Axon 

Axon is a tiny Deep Learning Library that uses MLIR to optimize tensor operations.

```python
import axon

@axon.jit
def mse_loss(y_pred: axon.Tensor, y_label: axon.Tensor) -> axon.Tensor:
    return 0.5 * (y_pred - y_label) ** 2
```
