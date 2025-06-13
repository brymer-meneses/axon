# Axon 

Axon is a tiny Deep Learning Library that uses MLIR to optimize the forward and
backward pass of the tensor operations.

```python
a = axon.Tensor(1)
b = axon.Tensor(1)
c = a + b
c.backward()

print(a.grad)
print(b.grad)
```
