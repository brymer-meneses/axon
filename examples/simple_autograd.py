import axon

from axon import LoweringLevel, LoweringOps

opts = LoweringOps(LoweringLevel.Axon)


@axon.jit(opts)
def matmul(a, b):
    l = 2 * a * b
    l.backward()


a = axon.ones((2, 2), requires_grad=True)
b = axon.ones((5, 2, 2), requires_grad=True)

matmul(a, b)
print(matmul.dump_ir())

if opts.execute:
    print(a.grad)
    print(b.grad)
