"""Causal feature construction.

Every feature here is computed from a window of prices *ending at* the current
tick, inclusive. None of them can see forward. That property is not a convention
to be careful about — it is checked by a test that recomputes the entire matrix
through `MarketView` (which raises on forward access) and asserts the two agree
exactly. A vectorised feature that is off by one index in the permissive
direction is invisible in the numbers and fatal to the study, so it gets a proof
rather than a comment.

The feature set is deliberately small and conventional. On a signal this weak, a
large feature space mostly buys faster overfitting: more columns means more ways
to fit noise, and the purged CV then has to spend its statistical power
detecting that rather than measuring an edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..data.store import TickSeries

DEFAULT_WINDOWS: tuple[int, ...] = (10, 30, 60)


@dataclass
class FeatureSet:
    """Feature matrix plus the tick index each row was computed at."""

    names: list[str]
    matrix: np.ndarray  # (n_samples, n_features)
    index: np.ndarray  # tick index in the source series for each row

    def __len__(self) -> int:
        return len(self.index)

    def subset(self, rows: np.ndarray) -> "FeatureSet":
        return FeatureSet(list(self.names), self.matrix[rows], self.index[rows])


def _windows(prices: np.ndarray, w: int) -> np.ndarray:
    """(n-w+1, w) view; row j covers prices[j:j+w] and ends at index j+w-1."""
    return sliding_window_view(prices, w)


def feature_names(windows: tuple[int, ...] = DEFAULT_WINDOWS) -> list[str]:
    names: list[str] = []
    for w in windows:
        names += [f"logret_{w}", f"zscore_{w}", f"vol_{w}", f"upfrac_{w}", f"rangepos_{w}"]
    return names


def build_features(
    series: TickSeries, windows: tuple[int, ...] = DEFAULT_WINDOWS
) -> FeatureSet:
    """Compute all features at every tick with enough history behind it."""
    if not windows:
        raise ValueError("at least one window is required")
    if any(w < 2 for w in windows):
        raise ValueError("windows must be at least 2 ticks")

    prices = np.asarray(series.price, dtype=np.float64)
    n = len(prices)
    max_w = max(windows)
    if n < max_w + 1:
        raise ValueError(f"need more than {max_w} ticks, got {n}")
    if np.any(prices <= 0):
        raise ValueError("prices must be positive to take log returns")

    start = max_w - 1  # first index with a full window behind it
    index = np.arange(start, n, dtype=np.int64)
    cols: list[np.ndarray] = []

    logp = np.log(prices)

    for w in windows:
        win = _windows(prices, w)  # row j ends at index j+w-1
        # Rows for our index range: index i maps to row i-(w-1).
        rows = index - (w - 1)
        win = win[rows]

        first, last = win[:, 0], win[:, -1]
        mean = win.mean(axis=1)
        std = win.std(axis=1)
        lo, hi = win.min(axis=1), win.max(axis=1)

        cols.append(np.log(last / first))  # logret

        safe_std = np.where(std > 0, std, np.nan)
        z = (last - mean) / safe_std
        cols.append(np.nan_to_num(z, nan=0.0))  # zscore

        lw = _windows(logp, w)[rows]
        rets = np.diff(lw, axis=1)
        cols.append(rets.std(axis=1))  # vol

        cols.append((rets > 0).mean(axis=1))  # upfrac

        span = hi - lo
        safe_span = np.where(span > 0, span, np.nan)
        rp = (last - lo) / safe_span
        cols.append(np.nan_to_num(rp, nan=0.5))  # rangepos: mid when flat

    return FeatureSet(
        names=feature_names(windows),
        matrix=np.column_stack(cols),
        index=index,
    )


def compute_row(view, windows: tuple[int, ...] = DEFAULT_WINDOWS) -> np.ndarray:
    """Reference implementation for a single tick, via a `MarketView`.

    Exists to prove `build_features` is causal. `MarketView` physically cannot
    return future data, so agreement between the two is evidence the vectorised
    path is not reaching forward. Too slow for production use — that is what
    `build_features` is for.
    """
    out: list[float] = []
    for w in windows:
        h = view.history(w)
        if len(h) < w:
            raise ValueError(f"insufficient history for window {w}")
        first, last = float(h[0]), float(h[-1])
        mean, std = float(h.mean()), float(h.std())
        lo, hi = float(h.min()), float(h.max())

        out.append(float(np.log(last / first)))
        out.append((last - mean) / std if std > 0 else 0.0)

        rets = np.diff(np.log(h))
        out.append(float(rets.std()))
        out.append(float((rets > 0).mean()))

        out.append((last - lo) / (hi - lo) if hi > lo else 0.5)
    return np.array(out, dtype=np.float64)
