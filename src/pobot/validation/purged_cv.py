"""Purged, embargoed walk-forward cross-validation.

This is the highest-value component in the repository, because the bias it
removes is the specific reason retail trading bots backtest profitably and lose
money live.

The problem. A trade opened at t0 settles at t1. Two trades opened seconds apart
have almost entirely overlapping lives, so they resolve on nearly the same price
path and their outcomes are close to the same observation counted twice. Ordinary
k-fold cross-validation shuffles such samples across the train/test boundary, so
a training sample can overlap a test sample and carry its answer. The model then
scores well on data it has effectively already seen. With 60-second expiries and
signals every few seconds, nearly every sample is contaminated this way.

The fix, following Lopez de Prado's *Advances in Financial Machine Learning*:

1. Walk forward. Test folds always follow their training data in time. Training
   on the future to predict the past scores well and means nothing.
2. Purge. Drop training samples whose [t0, t1] interval overlaps the test
   window's span. Their outcomes are partly determined by price action inside
   the test period.
3. Embargo. Drop training samples for a further interval after the test window.
   Serial correlation in features means samples just after a test fold still
   carry information about it, even without literal overlap.

`PurgedWalkForward` needs both t0 and t1 for every sample — which is why
`LabelSet` carries `expiry_ts`. Without t1 there is no way to know what overlaps
what, and the leak becomes invisible rather than absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class Fold:
    """One train/test split, plus an audit trail of what purging removed."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    test_start_ms: int
    test_end_ms: int
    n_purged: int
    n_embargoed: int


class PurgedWalkForward:
    """Expanding- or rolling-window splits with purge and embargo.

    Args:
        n_splits: Number of test folds. The series is divided into `n_splits`
            contiguous test blocks, each preceded by its training data.
        embargo_pct: Embargo length as a fraction of total series duration.
            0.01 (default) is the usual starting point. Raise it when features
            have long lookbacks — a 200-period moving average keeps information
            alive far longer than the trade itself.
        expanding: True (default) trains on everything before the test window.
            False uses a fixed-length rolling window, appropriate when you
            believe the market regime shifts and old data is misleading.
        min_train: Minimum training samples for a fold to be yielded. Folds with
            less are skipped rather than silently trained on nothing.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        embargo_pct: float = 0.01,
        expanding: bool = True,
        min_train: int = 50,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0, 1)")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.expanding = expanding
        self.min_train = min_train

    def split(self, t0: np.ndarray, t1: np.ndarray) -> Iterator[Fold]:
        """Yield folds for samples with entry times `t0` and expiry times `t1`.

        Both arrays must be sorted ascending by `t0`.
        """
        t0 = np.asarray(t0, dtype=np.int64)
        t1 = np.asarray(t1, dtype=np.int64)
        if len(t0) != len(t1):
            raise ValueError("t0 and t1 must be the same length")
        if len(t0) < self.n_splits:
            raise ValueError(f"need at least {self.n_splits} samples, got {len(t0)}")
        if np.any(np.diff(t0) < 0):
            raise ValueError("t0 must be sorted ascending")
        if np.any(t1 < t0):
            raise ValueError("every t1 must be >= its t0")

        n = len(t0)
        span = int(t0[-1] - t0[0]) or 1
        embargo_ms = int(span * self.embargo_pct)

        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)

        for k in range(self.n_splits):
            lo, hi = bounds[k], bounds[k + 1]
            if hi - lo <= 0:
                continue
            test_idx = np.arange(lo, hi)
            test_start = int(t0[lo])
            test_end = int(max(t1[lo:hi].max(), t0[hi - 1]))

            # Candidate training set: strictly before the test block. Samples
            # after the test block are never used, even in expanding mode —
            # that would be training on the future.
            cand = np.arange(0, lo)
            if not self.expanding:
                window = max(self.min_train, (hi - lo) * 2)
                cand = cand[-window:] if len(cand) > window else cand

            if len(cand) == 0:
                continue

            # Purge: drop training samples still live when the test block began.
            overlaps = t1[cand] >= test_start
            n_purged = int(overlaps.sum())
            cand = cand[~overlaps]

            # Embargo: drop training samples opening inside the embargo period
            # that follows the test block. Only bites when training data exists
            # after the test window, i.e. rolling mode near the series end.
            if embargo_ms > 0 and len(cand):
                emb = (t0[cand] > test_end) & (t0[cand] <= test_end + embargo_ms)
                n_embargoed = int(emb.sum())
                cand = cand[~emb]
            else:
                n_embargoed = 0

            if len(cand) < self.min_train:
                continue

            yield Fold(
                train_idx=cand,
                test_idx=test_idx,
                test_start_ms=test_start,
                test_end_ms=test_end,
                n_purged=n_purged,
                n_embargoed=n_embargoed,
            )


def overlap_fraction(t0: np.ndarray, t1: np.ndarray) -> float:
    """Average share of other samples each sample's life overlaps.

    A diagnostic worth running before trusting any significance test. A value
    near 0 means samples are effectively independent and n is close to your true
    sample size. A high value means your 10,000 "trades" carry the information
    of far fewer, and every p-value computed from them is optimistic.
    """
    t0 = np.asarray(t0, dtype=np.int64)
    t1 = np.asarray(t1, dtype=np.int64)
    n = len(t0)
    if n < 2:
        return 0.0
    # Samples are sorted by t0, so overlaps for i are the j>i with t0[j] < t1[i].
    j = np.searchsorted(t0, t1, side="left")
    counts = np.maximum(j - np.arange(n) - 1, 0)
    return float(counts.sum() * 2) / (n * (n - 1))
