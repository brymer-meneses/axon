import axon
from axon import Tensor


a = Tensor.ones((2, 5, 5), requires_grad=True, dtype=axon.float64)
b = Tensor.ones((5, 5), requires_grad=True, dtype=axon.float64)

c = a @ b
c.backward(Tensor.ones((2, 5, 5), dtype=axon.float64))

print(a.grad)
print(b.grad)
