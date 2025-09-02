import axon

from axon import LoweringOps

import numpy as np

opts = LoweringOps(LoweringOps.Level.Standard)
opts.level = LoweringOps.Level.Standard

@axon.jit()
def matmul(a, b): 
    l = a @ b
    l.backward()

a = axon.tensor(np.ones((2, 2), dtype=np.float32), requires_grad=True)
b = axon.tensor(np.ones((5, 2, 2), dtype=np.float32), requires_grad=True)

matmul(a,b)
print(matmul.dump_ir())
#
print(a.grad)
print(b.grad)

