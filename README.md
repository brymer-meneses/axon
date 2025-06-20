# Axon 

Axon is a tiny Deep Learning Library that uses MLIR to optimize tensor operations.

```python
import axon

ctx = axon.Context()

@ctx.compile
def linear(x: axon.Tensor, b: axon.Tensor) -> None:
    pass

axon.set_current_context(ctx)

t1 = axon.tensor(1, requires_grad=True)
t2 = axon.tensor(2, requires_grad=True)
t3 = t1 + t2
t3.backward()
```
