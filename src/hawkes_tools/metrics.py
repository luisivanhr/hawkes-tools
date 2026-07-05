"""Selection metrics for sparse-support recovery."""

from __future__ import annotations

import numpy as np

__all__ = ["support_fdp", "support_recall"]


def support_fdp(x_truth, x, eps: float = 1e-8) -> float:
    """False discovery proportion for support recovery."""

    selected = np.abs(x) > eps
    truth = np.abs(x_truth) > eps
    false_positives = np.logical_and(np.logical_not(truth), selected).sum()
    selected_count = max(selected.sum(), 1)
    return float(false_positives / selected_count)


def support_recall(x_truth, x, eps: float = 1e-8) -> float:
    """Recall of the true support."""

    selected = np.abs(x) > eps
    truth = np.abs(x_truth) > eps
    true_positives = np.logical_and(truth, selected).sum()
    return float(true_positives / truth.sum())
