"""Point-forecast metrics. All accept 1-D numpy arrays of equal length."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_pred - y_true).mean())


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float((np.abs(y_pred - y_true) / denom).mean() * 100.0)


def wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.abs(y_pred - y_true).sum() / (np.abs(y_true).sum() + eps) * 100.0)
