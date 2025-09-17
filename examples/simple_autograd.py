import axon
from axon import Tensor


a = Tensor.randn((2, 5, 5), requires_grad=True, dtype=axon.float64)
b = Tensor.randn((5, 5), requires_grad=True, dtype=axon.float64)

c = a @ b
c.backward(Tensor.ones((5, 5), dtype=axon.float64))

# print(a)
# print(b)
#
# print(c)
# print(c)

print(a.grad)
print(b.grad)
