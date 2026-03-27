"""Verify walk-forward splitter has no future data leakage."""
import numpy as np
import pytest

from src.preprocessing.splitter import WalkForwardSplitter


class TestWalkForwardSplitter:
    def test_train_always_before_val(self):
        """Core leakage test: max(train_idx) must always be < min(val_idx)."""
        splitter = WalkForwardSplitter(initial_train_size=100, step_size=10, val_size=20)
        for train_idx, val_idx in splitter.split(300):
            assert train_idx.max() < val_idx.min(), (
                f"Leakage! train max={train_idx.max()}, val min={val_idx.min()}"
            )

    def test_val_size_correct(self):
        splitter = WalkForwardSplitter(initial_train_size=100, step_size=10, val_size=20)
        for _, val_idx in splitter.split(300):
            assert len(val_idx) == 20

    def test_expanding_window_grows(self):
        splitter = WalkForwardSplitter(
            initial_train_size=100, step_size=10, val_size=20, expanding_window=True
        )
        train_sizes = [len(t) for t, _ in splitter.split(300)]
        assert train_sizes == sorted(train_sizes), "Expanding window train sizes should be non-decreasing"

    def test_rolling_window_constant(self):
        splitter = WalkForwardSplitter(
            initial_train_size=100, step_size=10, val_size=20, expanding_window=False
        )
        train_sizes = [len(t) for t, _ in splitter.split(300)]
        assert all(s == train_sizes[0] for s in train_sizes), \
            "Rolling window train sizes should be constant"

    def test_no_same_fold_leakage(self):
        """
        Within each fold, no train index should appear in val (same fold).
        Note: val from fold N legitimately enters train for fold N+1 in
        expanding-window CV — that is correct behaviour, not leakage.
        """
        splitter = WalkForwardSplitter(initial_train_size=100, step_size=10, val_size=20)
        for train_idx, val_idx in splitter.split(300):
            train_set = set(train_idx)
            val_set = set(val_idx)
            assert train_set.isdisjoint(val_set), (
                "Same-fold leakage: train and val indices overlap within a single fold"
            )

    def test_n_splits_count(self):
        splitter = WalkForwardSplitter(initial_train_size=100, step_size=10, val_size=20)
        n = splitter.n_splits(300)
        actual = sum(1 for _ in splitter.split(300))
        assert n == actual

    def test_insufficient_data(self):
        """Should yield 0 splits when data is smaller than initial_train + val."""
        splitter = WalkForwardSplitter(initial_train_size=100, step_size=10, val_size=20)
        splits = list(splitter.split(50))
        assert len(splits) == 0

    def test_indices_within_bounds(self):
        n = 250
        splitter = WalkForwardSplitter(initial_train_size=100, step_size=10, val_size=20)
        for train_idx, val_idx in splitter.split(n):
            assert train_idx.min() >= 0
            assert val_idx.max() < n
