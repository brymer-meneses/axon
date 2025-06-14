
from axon._axon_cpp import TensorId, Context

ctx = Context("hi")
t1 = ctx.declare_tensor(True, [5, 10, 10])
t2 = ctx.declare_tensor(True, [5, 10, 10])

ctx.record_batch_matmul(t1, t2)
