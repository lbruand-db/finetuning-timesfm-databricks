import numpy as np
import pytest

from src.metrics import mae, smape, wape


def test_mae_zero_when_equal():
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert mae(y, y.copy()) == pytest.approx(0.0)


def test_mae_known_diff():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 4.0, 6.0])
    # absolute errors: 1, 2, 3 → mean 2.0
    assert mae(y_true, y_pred) == pytest.approx(2.0)


def test_wape_known_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 22.0, 33.0])
    # sum |err| = 1+2+3 = 6; sum |y_true| = 60; → 10.0%
    assert wape(y_true, y_pred) == pytest.approx(10.0, abs=1e-4)


def test_wape_handles_all_zero_truth():
    # When y_true is all zeros, eps prevents division by zero.
    y_true = np.zeros(5)
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = wape(y_true, y_pred)
    assert np.isfinite(result)
    assert result > 0


def test_smape_zero_when_equal():
    y = np.array([1.0, 2.0, 3.0])
    assert smape(y, y.copy()) == pytest.approx(0.0, abs=1e-4)


def test_smape_bounded_at_200():
    # sMAPE peaks at 200% when y_true and y_pred have opposite signs.
    y_true = np.array([1.0, 1.0])
    y_pred = np.array([-1.0, -1.0])
    assert smape(y_true, y_pred) == pytest.approx(200.0, abs=1e-4)


def test_smape_handles_zeros():
    # eps prevents NaN when both true and pred are zero.
    y_true = np.zeros(3)
    y_pred = np.zeros(3)
    assert np.isfinite(smape(y_true, y_pred))
