import axon
from axon import Tensor, LoweringLevel

a = Tensor.ones((2, 5, 5), requires_grad=True)
b = Tensor.ones((5, 5), requires_grad=True)
grad = Tensor.ones((2, 5, 5))

c = a @ b + a

axon.inspect_ir(c, LoweringLevel.Axon)

print(a.grad)
print(b.grad)
