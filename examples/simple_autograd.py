# incomplete for now :<

import axon
import numpy as np

@axon.jit()
def mul(a, b): 
    l = a * b
    l.backward()


a = axon.tensor(5 * np.ones((3, 3), dtype=np.float32), requires_grad=True)
b = axon.tensor(4 * np.ones((3, 3), dtype=np.float32), requires_grad=True)

print(a.grad)
print(b.grad)

mul(a,b)

print(a.grad)
print(b.grad)

