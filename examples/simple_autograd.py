# incomplete for now :<

import axon

from axon import LoweringOps

@axon.jit()
def mul(a, b): 
    l = a * b
    l.backward()


a = axon.tensor([1, 2, 3], requires_grad=True)
b = axon.tensor([4, 5, 6], requires_grad=True)
print(a)
print(b)

mul(a,b)

print(a)
print(b)

