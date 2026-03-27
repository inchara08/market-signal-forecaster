"""Walk-forward cross-validation splitter for time series data."""
from __future__ import annotations
import numpy as np
from typing import Iterator


class WalkForwardSplitter:
    """
    Expanding-window walk-forward splitter.

    Yields (train_indices, val_indices) pairs where train always precedes val.
    Never shuffles — preserves temporal ordering.
    """

    def __init__(
        self,
        initial_train_size: int = 504,
        step_size: int = 21,
        val_size: int = 63,
        expanding_window: bool = True,
    ):
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.val_size = val_size
        self.expanding_window = expanding_window

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, val_idx) arrays."""
        start = 0
        train_end = self.initial_train_size

        while train_end + self.val_size <= n_samples:
            val_end = train_end + self.val_size

            if self.expanding_window:
                train_idx = np.arange(start, train_end)
            else:
                train_start = max(0, train_end - self.initial_train_size)
                train_idx = np.arange(train_start, train_end)

            val_idx = np.arange(train_end, val_end)
            yield train_idx, val_idx

            train_end += self.step_size

    def n_splits(self, n_samples: int) -> int:
        count = 0
        for _ in self.split(n_samples):
            count += 1
        return count
