import axon

from axon import LoweringLevel, LoweringOps

import numpy as np

opts = LoweringOps(LoweringLevel.Axon)

@axon.jit(opts)
def matmul(a, b): 
    l = a @ b + b 
    l.backward()

a = axon.ones((2, 2), dtype=axon.float32, requires_grad=True)
b = axon.ones((5, 2, 2), dtype=axon.float32, requires_grad=True)

matmul(a,b)
print(matmul.dump_ir())
#
print(a.grad)
print(b.grad)

