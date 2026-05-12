"""Lightweight tests for TimesFMInverterModel.

We don't exercise the actual TimesFM forward pass — that would require
the full ~400 MB checkpoint and a GPU-ish runtime. Instead we cover:

1. Input validation in `predict()` (missing column, short context).
2. The `_read_text` static helper.
"""
import numpy as np
import pandas as pd
import pytest

from src.model import TimesFMInverterModel


def _model_under_test(context_len: int = 8, horizon_len: int = 4) -> TimesFMInverterModel:
    """Construct a model wrapper with the attributes that `load_context`
    would normally populate, minus the heavy ML model itself."""
    m = TimesFMInverterModel()
    m.context_len = context_len
    m.horizon_len = horizon_len
    m.device = "cpu"
    return m


def test_predict_missing_context_column_raises():
    m = _model_under_test()
    bad_input = pd.DataFrame({"not_context": [[0.0] * 8]})
    with pytest.raises(ValueError, match="must have a 'context' column"):
        m.predict(context=None, model_input=bad_input)


def test_predict_too_short_context_raises():
    m = _model_under_test(context_len=10)
    short_ctx = pd.DataFrame({"context": [[0.0] * 5]})
    with pytest.raises(ValueError, match="at least 10 points"):
        m.predict(context=None, model_input=short_ctx)


def test_read_text_strips_trailing_newline(tmp_path):
    f = tmp_path / "id.txt"
    f.write_text("google/timesfm-2.5-200m-transformers\n")
    # `_read_text` returns the raw bytes; load_context calls .strip() on it.
    raw = TimesFMInverterModel._read_text(str(f))
    assert raw.rstrip() == "google/timesfm-2.5-200m-transformers"
