from axon.datasets import DatasetIterator
import numpy as np

from axon import nn, Tensor, optim, LoweringLevel

import axon.nn.functional as F
import axon


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
        l1_out = F.relu(self.l1(x))
        l2_out = self.l2(l1_out)
        return l2_out


def train(
    model: Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = 500,
    epochs: int = 4,
    lr: float = 5e-2,
    inspect_ir: bool = False,
):
    """
    Train the MNIST model on the provided training data.

    If x_test and y_test are provided, also evaluate the cross-entropy loss
    on the test set at the end of each epoch.

    Returns:
        (model, last_train_loss, last_test_loss)
    """
    x_train = (x_train / 255.0).astype(np.float32)

    train_dataset = DatasetIterator(
        x_train, y_train, batch_size=batch_size, shuffle=True
    )

    # Use SGD with a mild step-wise LR decay: every 100 steps, lr *= 0.99
    sgd = optim.SGD(model.parameters(), lr=lr, gamma=0.50, decay_every=100, min_lr=1e-4)

    total_epoch_loss = [0.0] * epochs
    step = 0

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_dataset():
            sgd.zero_grad()

            inputs = x_batch.reshape(-1, 784)
            inputs = Tensor(inputs)

            targets = as_one_hot(10, y_batch).astype(np.float32)
            logits = model(inputs)

            loss = F.cross_entropy(logits, Tensor(targets))
            loss.backward()

            if inspect_ir and epoch == 0 and step == 0:
                axon.inspect_ir(loss, LoweringLevel.Axon)

            sgd.step()

            step += 1

            if step % 10 == 0:
                print(f"step={step:5d} lr={sgd.lr:.5f} loss={loss.item():.6f}")

            total_loss += loss.item()

        total_epoch_loss[epoch] = total_loss

    return total_epoch_loss


def evaluate(
    model: Model, x_test: np.ndarray, y_test: np.ndarray, batch_size: int = 100
):
    """Evaluate model on test set and return average cross-entropy loss (float)."""
    x_test = (x_test / 255.0).astype(np.float32)
    test_dataset = DatasetIterator(x_test, y_test, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    batches = 0

    with axon.no_grad():
        for x_batch, y_batch in test_dataset():
            inputs = Tensor(x_batch.reshape(-1, 784))
            targets = Tensor(as_one_hot(10, y_batch).astype(np.float32))
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            total_loss += loss.item()
            batches += 1

    if batches == 0:
        return None

    return total_loss / batches


def main() -> None:
    (x_train, y_train), (x_test, y_test) = load_mnist_dataset()

    model = Model()

    train_loss = train(
        model,
        x_train,
        y_train,
        batch_size=500,
        epochs=4,
        lr=5e-1,
        inspect_ir=True,
    )

    test_loss = evaluate(
        model,
        x_test,
        y_test,
        batch_size=100,
    )

    print(f"Final train loss (per-epoch sums): {train_loss}")
    print(f"Final test  loss (avg): {test_loss}")


if __name__ == "__main__":
    main()
