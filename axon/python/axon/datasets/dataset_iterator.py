from typing import Iterator, NamedTuple


import numpy as np


class Batch(NamedTuple):
    inputs: np.ndarray
    targets: np.ndarray


class DatasetIterator:
    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self) -> Iterator[Batch]:
        assert len(self.inputs) == len(self.targets), (
            "inputs and targets must have the same length"
        )

        starting_points = np.arange(0, len(self.inputs), self.batch_size)

        if self.shuffle:
            np.random.shuffle(starting_points)

        for start in starting_points:
            end = start + self.batch_size

            batch_inputs = self.inputs[start:end]
            batch_targets = self.targets[start:end]

            yield Batch(batch_inputs, batch_targets)
