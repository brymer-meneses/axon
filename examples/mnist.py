from axon.datasets import DatasetIterator
import numpy as np
import axon

from axon import nn, Tensor, builtin, optim


def load_mnist_dataset():
    import os
    from urllib import request

    url = r"https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = ".datasets/mnist-dataset.npz"

    if not os.path.exists(path):
        dir = os.path.dirname(path)
        os.makedirs(dir, exist_ok=True)
        request.urlretrieve(url, path)

    with np.load(path) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    return (x_train, y_train), (x_test, y_test)


def as_one_hot(num_classes: int, input: np.ndarray) -> np.ndarray:
    return np.identity(num_classes)[input]


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.l1 = nn.Linear(28 * 28, 256)
        self.l2 = nn.Linear(256, 10)

    def forward(self, x: Tensor) -> Tensor:
        l1_out = self.l1(x).relu()
        l2_out = self.l2(l1_out)
        # Return raw logits; cross-entropy applies softmax internally
        return l2_out


(x_train, y_train), (x_test, y_test) = load_mnist_dataset()

x_train = (x_train / 255.0).astype(np.float32)

train_dataset = DatasetIterator(x_train, y_train, batch_size=500, shuffle=True)
test_dataset = DatasetIterator(x_train, y_train, batch_size=100, shuffle=True)

model = Model()
optim = optim.SGD(model.parameters(), lr=5e-2)


for x_batch, y_batch in train_dataset():
    optim.zero_grad()

    inputs = x_batch.reshape(-1, 784)
    inputs = Tensor(inputs)

    targets = as_one_hot(10, y_batch).astype(np.float32)
    logits = model(inputs)

    loss = builtin.cross_entropy(logits, Tensor(targets))
    loss.backward()

    optim.step()

    print(loss)
