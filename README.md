# Axon 

 Axon is a lightweight deep learning library that lowers tensor graphs to MLIR
 and JIT‑compiles them for fast execution. 

## Building

You need to have `uv`, `cmake`, `make` and a compiler that supports C++23. 

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
        self.B = Tensor.randn((out_features, ), requires_grad=True)

    def forward(self, x):
        return x @ self.W + self.B


model = Model(100, 100)
optim = optim.SGD(model.parameters(), lr=0.01)

for (inputs, targets) in dataset:
    optim.zero_grad()

    preds = model(inputs)
    l = builtin.mse_loss(targets, preds)
    l.backward()

    optim.step()
```
