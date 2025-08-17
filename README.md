# Axon 

Axon is a tiny Deep Learning Library that uses MLIR to optimize tensor operations.

```python
import axon

def mse_loss(y, y_hat):
    return 0.5 * (y-y_hat) ** 2

class Model(axon.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = axon.rand((in_features, out_features))
        self.B = axon.rand((out_features, 1))

    def forward(self, x):
        return self.W @ x + self.B
        
ctx = axon.Context()
x = ctx.tensor([...], requires_grad=True)
y = ctx.tensor([...])

model = Model()

y_hat = model(x)
l = mse_loss(y, y_hat)
ctx.backward(l)
```
