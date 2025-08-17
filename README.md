# Axon 

Axon is a tiny Deep Learning Library that uses MLIR to optimize tensor operations.

```python
import axon

def mse_loss(y, y_hat):
    return 0.5 * (y-y_hat) ** 2

class Model(axon.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = axon.rand((in_features, out_features), requires_grad=True)
        self.B = axon.rand((out_features, 1), requires_grad=True)

    def forward(self, x):
        return x @ self.W + self.B
        
x = axon.tensor([...])
y = axon.tensor([...])
model = Model(512, 1024)

# Having the `axon.Context` resource manager active will trace every tensor
# operation. And compile these operations in a MLIR Module.
with axon.Context() as ctx:
    y_hat = model(x)
    l = mse_loss(y, y_hat)
    l.backward()
```
