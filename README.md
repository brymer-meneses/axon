# Axon 

Axon is a tiny Deep Learning Library that uses MLIR to optimize tensor operations.

```python
a = axon.Tensor(1)
b = axon.Tensor(1)
c = a + b
c.backward()

print(a.grad)
print(b.grad)
```
