"""Window datasets for TimesFM 2.5 fine-tuning.

Mirrors the upstream `finetune_lora.py` design: pre-sample random
(series, start) windows for training, last-window per series for
validation. No external normalisation — TimesFM handles RevIN
internally, so we feed raw values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def series_from_silver(
    pdf: pd.DataFrame,
    series_id_col: str,
    target_col: str,
    ts_col: str = "ts",
) -> tuple[list[str], list[np.ndarray]]:
    """Group a long-format DataFrame into one float32 array per series.

    Sorts by timestamp inside each group and drops series with any NaN
    in the target. Returns (series_ids, series_arrays) in matched order.
    """
    ids: list[str] = []
    arrays: list[np.ndarray] = []
    for sid, group in pdf.groupby(series_id_col, sort=True):
        ordered = group.sort_values(ts_col)[target_col].to_numpy(dtype=np.float32)
        if np.isnan(ordered).any():
            continue
        ids.append(str(sid))
        arrays.append(ordered)
    return ids, arrays


class TimeSeriesRandomWindowDataset(Dataset):
    """Random-window dataset: each item is a (context, target) pair sliced
    from one of the input series at a uniformly-sampled start position.

    Pre-samples `num_samples` (series_idx, start) tuples up front so the
    dataset is index-stable across DataLoader workers.
    """

    def __init__(
        self,
        series_list: list[np.ndarray],
        context_len: int,
        horizon_len: int,
        num_samples: int = 5000,
        seed: int = 42,
    ):
        self.series_list = series_list
        self.context_len = context_len
        self.horizon_len = horizon_len

        min_len = context_len + horizon_len
        valid = [i for i, s in enumerate(series_list) if len(s) >= min_len]
        if not valid:
            raise ValueError(
                f"No series long enough for context_len={context_len} + "
                f"horizon_len={horizon_len}. Shortest series: "
                f"{min(len(s) for s in series_list)}"
            )

        rng = np.random.default_rng(seed)
        self.samples: list[tuple[int, int]] = []
        for _ in range(num_samples):
            idx = int(rng.choice(valid))
            series = series_list[idx]
            max_start = len(series) - min_len
            start = int(rng.integers(0, max_start + 1))
            self.samples.append((idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        idx, start = self.samples[index]
        series = self.series_list[idx]
        end = start + self.context_len + self.horizon_len
        context = torch.tensor(series[start : start + self.context_len], dtype=torch.float32)
        target = torch.tensor(series[start + self.context_len : end], dtype=torch.float32)
        return context, target


class TimeSeriesLastWindowDataset(Dataset):
    """One (context, target) pair per series, taken from the tail."""

    def __init__(
        self,
        series_list: list[np.ndarray],
        context_len: int,
        horizon_len: int,
    ):
        min_len = context_len + horizon_len
        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []
        for s in series_list:
            if len(s) >= min_len:
                ctx = torch.tensor(s[-min_len:-horizon_len], dtype=torch.float32)
                tgt = torch.tensor(s[-horizon_len:], dtype=torch.float32)
                self.items.append((ctx, tgt))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]
