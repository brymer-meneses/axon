# Axon 

Axon is a tiny Deep Learning Library that uses MLIR to optimize tensor operations.

```python
import axon

@axon.jit
def mse_loss(y_pred: axon.Tensor, y_label: axon.Tensor) -> axon.Tensor:
    return 0.5 * (y_pred - y_label) ** 2

@axon.jit
def func(a, b):
 return a + b

def module(func):

    def decorated(a, b):
        a = module.declare_param(a.requires_grad)
        b = module.declare_param(b.requires_grad)

        result = func(a, b)

        module.build_backward(result)

        module.compile()

        return module.forward
    
    return decorated
```
