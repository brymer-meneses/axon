import axon

from axon import Tensor


a = Tensor([1, 2, 3], dtype=axon.float32)
b = Tensor([1, 2, 3], dtype=axon.float32)

for i in range(0, 10):
    c = a + b
    print(c)
