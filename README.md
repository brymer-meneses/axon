# Axon 

Axon is a tiny Deep Learning Library that uses MLIR to optimize tensor operations.

```python
import axon

jit = axon.create_jit_context()

@jit.compile
class Linear(axon.Module):

    def __init__(self) -> None:
        super().__init__()
        self.w = axon.rand((512, 1024), requires_grad=True)
        self.b = axon.rand((1024, 1), requires_grad=True)

    def forward(self, x: axon.Tensor) -> axon.Tensor:
        return self.w @ x + self.b

@jit.compile
def mse_loss(y_pred: axon.Tensor, y_label: axon.Tensor) -> axon.Tensor:
    return 0.5 * (y_pred - y_label) ** 2
```
