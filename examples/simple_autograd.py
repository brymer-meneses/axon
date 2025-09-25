import axon
from axon import Tensor, LoweringLevel

a = Tensor.ones((2, 5, 5), requires_grad=True)
b = Tensor.ones((5, 5), requires_grad=True)

c = (a @ b) ** 2
l = c.sum()
l.backward()


print(a.grad)
print(b.grad)
