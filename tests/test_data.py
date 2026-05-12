import numpy as np
import pandas as pd
import pytest

from src.data import (
    TimeSeriesLastWindowDataset,
    TimeSeriesRandomWindowDataset,
    series_from_silver,
)

# ---------- series_from_silver -------------------------------------------


def _silver(rows):
    return pd.DataFrame(rows, columns=["inverter_id", "ts", "ac_power_kw"])


def test_series_from_silver_groups_by_id():
    pdf = _silver(
        [
            ("A", 0, 1.0),
            ("A", 1, 2.0),
            ("A", 2, 3.0),
            ("B", 0, 10.0),
            ("B", 1, 20.0),
        ]
    )
    ids, arrs = series_from_silver(pdf, "inverter_id", "ac_power_kw")
    assert ids == ["A", "B"]
    np.testing.assert_array_equal(arrs[0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(arrs[1], [10.0, 20.0])


def test_series_from_silver_sorts_by_ts():
    # Rows arrive out of timestamp order.
    pdf = _silver(
        [
            ("A", 2, 3.0),
            ("A", 0, 1.0),
            ("A", 1, 2.0),
        ]
    )
    _, arrs = series_from_silver(pdf, "inverter_id", "ac_power_kw")
    np.testing.assert_array_equal(arrs[0], [1.0, 2.0, 3.0])


def test_series_from_silver_drops_nan():
    pdf = _silver(
        [
            ("A", 0, 1.0),
            ("A", 1, np.nan),
            ("A", 2, 3.0),
            ("B", 0, 10.0),
            ("B", 1, 20.0),
        ]
    )
    ids, arrs = series_from_silver(pdf, "inverter_id", "ac_power_kw")
    # A had a NaN → dropped entirely
    assert ids == ["B"]
    np.testing.assert_array_equal(arrs[0], [10.0, 20.0])


def test_series_from_silver_returns_float32():
    pdf = _silver([("A", 0, 1.0), ("A", 1, 2.0)])
    _, arrs = series_from_silver(pdf, "inverter_id", "ac_power_kw")
    assert arrs[0].dtype == np.float32


# ---------- TimeSeriesRandomWindowDataset --------------------------------


def _series_pool(lengths):
    return [np.arange(n, dtype=np.float32) for n in lengths]


def test_random_window_len_matches_num_samples():
    ds = TimeSeriesRandomWindowDataset(
        _series_pool([100, 100]), context_len=10, horizon_len=5, num_samples=42, seed=0
    )
    assert len(ds) == 42


def test_random_window_item_shapes():
    ds = TimeSeriesRandomWindowDataset(
        _series_pool([100]), context_len=10, horizon_len=5, num_samples=4, seed=0
    )
    ctx, tgt = ds[0]
    assert ctx.shape == (10,)
    assert tgt.shape == (5,)


def test_random_window_context_target_contiguous():
    # The target should start exactly where the context ends.
    series = np.arange(100, dtype=np.float32)
    ds = TimeSeriesRandomWindowDataset(
        [series], context_len=10, horizon_len=5, num_samples=8, seed=7
    )
    for i in range(len(ds)):
        ctx, tgt = ds[i]
        # ctx and tgt are slices of np.arange, so target[0] = context[-1] + 1
        assert float(tgt[0]) == float(ctx[-1]) + 1.0


def test_random_window_deterministic_with_seed():
    pool = _series_pool([100, 100])
    a = TimeSeriesRandomWindowDataset(pool, 10, 5, num_samples=20, seed=42)
    b = TimeSeriesRandomWindowDataset(pool, 10, 5, num_samples=20, seed=42)
    assert a.samples == b.samples


def test_random_window_different_seeds_diverge():
    pool = _series_pool([200, 200])
    a = TimeSeriesRandomWindowDataset(pool, 10, 5, num_samples=50, seed=1)
    b = TimeSeriesRandomWindowDataset(pool, 10, 5, num_samples=50, seed=2)
    assert a.samples != b.samples


def test_random_window_raises_when_no_series_long_enough():
    # Both series are shorter than context_len + horizon_len = 30.
    with pytest.raises(ValueError, match="No series long enough"):
        TimeSeriesRandomWindowDataset(
            _series_pool([10, 20]), context_len=20, horizon_len=10, num_samples=5
        )


def test_random_window_only_picks_eligible_series():
    # One eligible series, one too short → every sample must come from idx 0.
    pool = [np.arange(100, dtype=np.float32), np.arange(5, dtype=np.float32)]
    ds = TimeSeriesRandomWindowDataset(pool, 10, 5, num_samples=30, seed=0)
    assert all(idx == 0 for idx, _ in ds.samples)


# ---------- TimeSeriesLastWindowDataset ----------------------------------


def test_last_window_one_item_per_long_series():
    pool = _series_pool([100, 100, 100])
    ds = TimeSeriesLastWindowDataset(pool, context_len=10, horizon_len=5)
    assert len(ds) == 3


def test_last_window_skips_short_series():
    pool = [
        np.arange(100, dtype=np.float32),
        np.arange(5, dtype=np.float32),  # too short
        np.arange(50, dtype=np.float32),
    ]
    ds = TimeSeriesLastWindowDataset(pool, context_len=10, horizon_len=5)
    assert len(ds) == 2


def test_last_window_uses_tail():
    series = np.arange(100, dtype=np.float32)
    ds = TimeSeriesLastWindowDataset([series], context_len=10, horizon_len=5)
    ctx, tgt = ds[0]
    # tgt should be the final 5 values
    np.testing.assert_array_equal(tgt.numpy(), np.arange(95, 100, dtype=np.float32))
    # ctx is the 10 values immediately before
    np.testing.assert_array_equal(ctx.numpy(), np.arange(85, 95, dtype=np.float32))
