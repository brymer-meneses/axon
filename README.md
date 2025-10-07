# Axon

Axon is a lightweight deep learning library that lowers tensor graphs to MLIR
and JIT‑compiles them for fast execution using LLVM. It is lazy by default, and
has a familiar PyTorch API.


## Getting Started

Requirements: `uv`, `cmake`, `make`, and a compiler with C++23 support.

```bash
# create an isolated environment
uv venv
# build and install Axon
make install
# try the MNIST example
uv run examples/mnist.py
```

## Example: MNIST
This condensed version of `examples/mnist.py`.

```python
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

import axon
from axon import Tensor, nn, optim, LoweringLevel
from axon.datasets import DatasetIterator
import axon.nn.functional as F

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

def load_mnist():
    path = Path(".datasets/mnist.npz")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(MNIST_URL, path)
    with np.load(path) as data:
        return (data["x_train"], data["y_train"]), (data["x_test"], data["y_test"])


def as_one_hot(labels: np.ndarray) -> np.ndarray:
    return np.eye(10, dtype=np.float32)[labels]


class MnistMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 10)

    def forward(self, x: Tensor) -> Tensor:
        return self.l3(F.relu(self.l2(F.relu(self.l1(x)))))


(x_train, y_train), _ = load_mnist()
x_train = (x_train / 255.0).astype(np.float32)
train = DatasetIterator(x_train, y_train, batch_size=512, shuffle=True)

model = MnistMLP()
sgd = optim.SGD(model.parameters(), lr=5e-2, gamma=0.5, decay_every=100, min_lr=1e-4)

step = 0
for (x_batch, y_batch) in train():

    inputs = Tensor(x_batch.reshape(-1, 28 * 28))
    targets = Tensor(as_one_hot(y_batch))

    logits = model(inputs)  
    loss = F.cross_entropy(logits, targets)

    if step == 0:
        print(loss)  
        axon.inspect_ir(loss, LoweringLevel.Axon)  

    sgd.zero_grad()
    loss.backward()  
    sgd.step()

    step += 1

    if step % 20 == 0:
        print(f"step={step:03d} loss={loss.item():.4f}")

    if step == 60:
        break
```

This is then lowered into the following IR, which represents the whole forward and backward graph.

```mlir
module {
  func.func @graph(%arg0: memref<500x256xf32>, %arg1: memref<500x256xf32>, %arg2: memref<256x10xf32>, %arg3: memref<256x10xf32>, %arg4: memref<10xf32>, %arg5: memref<10xf32>, %arg6: memref<500x10xf32>) -> tensor<f32> {
    %0 = axon.constant dense<1.000000e+00> : tensor<f32>
    %1 = bufferization.to_tensor %arg0 restrict : memref<500x256xf32> to tensor<500x256xf32>
    %2 = bufferization.to_tensor %arg2 restrict : memref<256x10xf32> to tensor<256x10xf32>
    %3 = axon.matmul %1, %2 : tensor<500x256xf32>, tensor<256x10xf32> -> tensor<500x10xf32>
    %4 = bufferization.to_tensor %arg4 restrict : memref<10xf32> to tensor<10xf32>
    %5 = axon.reshape %4, tensor<10xf32> -> tensor<1x10xf32> {target_shape = array<i64: 1, 10>}
    %6 = axon.expand_dims %5 : tensor<1x10xf32> -> tensor<500x10xf32> {mappings = [[0, 500]]}
    %7 = axon.add %3, %6 -> tensor<500x10xf32>
    %8 = axon.softmax %7, 1 : tensor<500x10xf32> -> tensor<500x10xf32>
    %9 = math.log %8 : tensor<500x10xf32>
    %10 = bufferization.to_tensor %arg6 restrict : memref<500x10xf32> to tensor<500x10xf32>
    %11 = axon.mul %10, %9 -> tensor<500x10xf32>
    %12 = axon.sum %11, 1 : tensor<500x10xf32> -> tensor<500xf32> {keep_dims = false}
    %13 = axon.neg %12 -> tensor<500xf32>
    %14 = axon.mean %13, 0 : tensor<500xf32> -> tensor<f32> {keep_dims = false}
    %15 = axon.unsqueeze %0 : tensor<f32> -> tensor<1xf32> {dim = 0 : i64}
    %16 = axon.expand_dims %15 : tensor<1xf32> -> tensor<500xf32> {mappings = [[0, 500]]}
    %17 = axon.scalar_mul %16 * 2.000000e-03 : f32 -> tensor<500xf32>
    %18 = axon.neg %17 -> tensor<500xf32>
    %19 = axon.unsqueeze %18 : tensor<500xf32> -> tensor<500x1xf32> {dim = 1 : i64}
    %20 = axon.expand_dims %19 : tensor<500x1xf32> -> tensor<500x10xf32> {mappings = [[1, 10]]}
    %21 = axon.mul %20, %10 -> tensor<500x10xf32>
    %22 = axon.pow %8 -> tensor<500x10xf32> {exponent = -1.000000e+00 : f64}
    %23 = axon.mul %21, %22 -> tensor<500x10xf32>
    %24 = axon.mul %23, %8 -> tensor<500x10xf32>
    %25 = axon.sum %24, 1 : tensor<500x10xf32> -> tensor<500x1xf32> {keep_dims = true}
    %26 = axon.expand_dims %25 : tensor<500x1xf32> -> tensor<500x10xf32> {mappings = [[1, 10]]}
    %27 = axon.sub %23, %26 -> tensor<500x10xf32>
    %28 = axon.mul %8, %27 -> tensor<500x10xf32>
    %29 = axon.sum %28, 0 : tensor<500x10xf32> -> tensor<1x10xf32> {keep_dims = true}
    %30 = axon.reshape %29, tensor<1x10xf32> -> tensor<10xf32> {target_shape = array<i64: 10>}
    %31 = axon.matmul %28, %2 : tensor<500x10xf32>, tensor<256x10xf32> -> tensor<500x256xf32> {transpose_rhs = true}
    %32 = axon.matmul %1, %28 : tensor<500x256xf32>, tensor<500x10xf32> -> tensor<256x10xf32> {transpose_lhs = true}
    %33 = bufferization.to_tensor %arg1 restrict writable : memref<500x256xf32> to tensor<500x256xf32>
    axon.accumulate %33, %31 : tensor<500x256xf32>, tensor<500x256xf32>
    %34 = bufferization.to_tensor %arg3 restrict writable : memref<256x10xf32> to tensor<256x10xf32>
    axon.accumulate %34, %32 : tensor<256x10xf32>, tensor<256x10xf32>
    %35 = bufferization.to_tensor %arg5 restrict writable : memref<10xf32> to tensor<10xf32>
    axon.accumulate %35, %30 : tensor<10xf32>, tensor<10xf32>
    return %14 : tensor<f32>
  }
}
```

