# Axon 

Axon is a tiny Deep Learning Library that uses MLIR to optimize tensor operations by
using just-in-time compilation to execute functions that involve tensors.

## Building

You need to have `uv`, `cmake` `make` and a compiler that supports C++23. 
```bash
# create a virtual environment
uv venv
# build and install the library
make install
# run the example!
uv run examples/mnist.py
```

## Example

```python
import axon
from axon import nn, optim, builtin, Tensor

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.W = Tensor.randn((in_features, out_features), requires_grad=True)
        self.B = Tensor.randn((out_features, 1), requires_grad=True)

    def forward(self, x):
        return x @ self.W + self.B


optim = optim.SGD(model.parameters(), lr=0.01)
model = Model()

for (x, y) in dataset:
    y_hat = model(x)
    l = builtin.mse_loss(y, y_hat)
    l.backward()

    optim.zero_grad()
    optim.step()
```
