# Axon 

Axon is a tiny Deep Learning Library that uses MLIR to optimize tensor operations.

```python
import axon

class Model(axon.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = axon.rand((in_features, out_features), requires_grad=True)
        self.B = axon.rand((out_features, 1), requires_grad=True)

    def forward(self, x):
        return x @ self.W + self.B

def mse_loss(y, y_hat):
    return 0.5 * (y-y_hat) ** 2

@axon.jit()
def training_step(x, y, model):
    y_hat = model(x)
    l = mse_loss(y, y_hat)
    l.backward()

optim = axon.SGD(model.parameters(), lr=0.01)
for (x, y) in dataset:
    l = training_step(x, y, model)
    optim.zero_grad()
    optim.step()
```
