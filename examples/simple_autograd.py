import axon

from axon import Tensor


a = Tensor([1, 2, 3, 4], dtype=axon.float32)
b = Tensor([1, 2, 3, 4], dtype=axon.float32)

c = a + b

print(c)
