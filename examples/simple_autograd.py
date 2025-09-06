import axon

from axon import LoweringLevel, LoweringOps

opts = LoweringOps(LoweringLevel.LLVM)


@axon.jit(opts)
def matmul(a, b):
    l = a @ b
    l.backward()


a = axon.ones((2, 2), requires_grad=True)
b = axon.ones((5, 2, 2), requires_grad=True)

matmul(a, b)
print(matmul.dump_ir())
#
print(a.grad)
print(b.grad)
