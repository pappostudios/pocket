"""Feed lag estimation.

This module tests the one structurally sound edge in binary options: whether the
broker's quote stream is a delayed copy of a real market. If it is, the reference
feed's *current* price partly determines the broker's *next* price, and the edge
is mechanical rather than statistical — you are not predicting the market, you
are reading a clock that runs slow.

Method. Resample both feeds onto a common time grid, difference to returns, and
cross-correlate at a range of offsets. Define

    corr(k) = correlation( broker_returns[t], reference_returns[t - k] )

A peak at k > 0 means the broker's moves *follow* the reference's by k
milliseconds — the broker feed lags. A peak at k = 0 means the two move together
and there is no exploitable delay. A peak at k < 0 would mean the broker leads,
which for a retail broker quoting a major pair means your reference feed is the
slow one, not that you have found free money.

Three cautions, in order of how much money they cost:

1. A lag is not a strategy. Spread, entry latency, minimum stake, and payout all
   eat the edge, and the broker's own latency to *your* order is on top. Feed the
   measured lag into `BinaryContract.entry_latency_ms` and backtest it properly
   before believing anything.

2. Brokers monitor for exactly this. Latency arbitrage is the named reason in
   most retail terms of service for voiding trades and closing accounts, and it
   is detectable from trade timing alone. Measuring the lag is research; trading
   it is a decision to make with the terms in front of you.

3. Correlation peaks appear in pure noise if you scan enough offsets. The
   reported significance applies a Šidák correction across every offset tested,
   for the same reason the strategy gate corrects across every variant tried.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

from ..data.store import TickSeries


@dataclass
class LagEstimate:
    """Result of a cross-correlation lag scan."""

    peak_lag_ms: int
    peak_corr: float
    zero_lag_corr: float
    n_samples: int
    grid_ms: int
    lags_ms: np.ndarray
    corrs: np.ndarray
    z_score: float
    p_value: float
    alpha_adjusted: float
    significant: bool
    notes: list[str]

    @property
    def broker_lags(self) -> bool:
        """True when the broker feed follows the reference — the exploitable case."""
        return self.significant and self.peak_lag_ms > 0

    def report(self) -> str:
        lines = [
            f"peak at {self.peak_lag_ms:+d}ms  r={self.peak_corr:.4f}  "
            f"(r={self.zero_lag_corr:.4f} at zero lag)",
            f"  samples         {self.n_samples} on a {self.grid_ms}ms grid",
            f"  z               {self.z_score:.2f}  p={self.p_value:.3g} "
            f"(alpha {self.alpha_adjusted:.3g}, corrected for "
            f"{len(self.lags_ms)} offsets)",
        ]
        if self.broker_lags:
            lines.append(
                f"  => broker feed follows the reference by ~{self.peak_lag_ms}ms."
            )
            lines.append(
                "     Backtest it with entry_latency_ms set realistically before "
                "believing it, and read the terms of service."
            )
        elif self.significant and self.peak_lag_ms < 0:
            lines.append(
                "  => broker leads the reference: your reference feed is the slow "
                "one. Not tradeable."
            )
        else:
            lines.append("  => no significant lag detected.")
        for n in self.notes:
            lines.append(f"  - {n}")
        return "\n".join(lines)


def resample_last(series: TickSeries, grid: np.ndarray) -> np.ndarray:
    """Last observed price at each grid point, NaN before the series starts.

    Last-at-or-before, never interpolated: interpolating toward the next tick
    would leak information the clock had not yet produced, and here that leak
    would manufacture exactly the lag the module is trying to measure.
    """
    idx = np.searchsorted(series.ts, grid, side="right") - 1
    out = np.full(len(grid), np.nan, dtype=np.float64)
    valid = idx >= 0
    out[valid] = series.price[idx[valid]]
    return out


def _log_returns(prices: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(prices))
    return r


def estimate_lag(
    broker: TickSeries,
    reference: TickSeries,
    *,
    grid_ms: int = 100,
    max_lag_ms: int = 3_000,
    alpha: float = 0.05,
) -> LagEstimate:
    """Estimate how far the broker feed trails the reference feed.

    Args:
        grid_ms: Resampling interval. Sets the resolution of the answer, so it
            must be no coarser than the lag you hope to detect.
        max_lag_ms: Half-width of the offset scan.
        alpha: Significance level before correction across offsets.
    """
    if grid_ms <= 0:
        raise ValueError("grid_ms must be positive")
    if max_lag_ms < grid_ms:
        raise ValueError("max_lag_ms must be at least grid_ms")
    if len(broker) < 2 or len(reference) < 2:
        raise ValueError("both feeds need at least two ticks")

    notes: list[str] = []

    start = max(int(broker.ts[0]), int(reference.ts[0]))
    end = min(int(broker.ts[-1]), int(reference.ts[-1]))
    if end - start < max_lag_ms * 4:
        raise ValueError(
            f"overlapping capture is only {(end - start) / 1000:.1f}s — too short "
            f"to scan +/-{max_lag_ms}ms of offsets"
        )

    grid = np.arange(start, end + 1, grid_ms, dtype=np.int64)
    b = resample_last(broker, grid)
    r = resample_last(reference, grid)

    rb, rr = _log_returns(b), _log_returns(r)
    ok = np.isfinite(rb) & np.isfinite(rr)
    rb, rr = rb[ok], rr[ok]
    if len(rb) < 100:
        raise ValueError(f"only {len(rb)} usable return pairs; capture more data")

    # A feed that barely moves on this grid carries no timing information, and
    # its correlations will be dominated by the handful of ticks that did move.
    if np.count_nonzero(rb) < len(rb) * 0.05:
        notes.append(
            "broker feed is nearly constant on this grid — try a coarser grid_ms"
        )
    if np.count_nonzero(rr) < len(rr) * 0.05:
        notes.append(
            "reference feed is nearly constant on this grid — try a coarser grid_ms"
        )

    steps = int(max_lag_ms // grid_ms)
    lags = np.arange(-steps, steps + 1, dtype=int)
    corrs = np.full(len(lags), np.nan, dtype=np.float64)

    for i, k in enumerate(lags):
        # corr(broker[t], reference[t - k]); k > 0 means broker follows reference.
        if k >= 0:
            x, y = rb[k:], rr[: len(rr) - k] if k else rr
        else:
            x, y = rb[: len(rb) + k], rr[-k:]
        if len(x) < 50 or np.std(x) == 0 or np.std(y) == 0:
            continue
        corrs[i] = float(np.corrcoef(x, y)[0, 1])

    if not np.any(np.isfinite(corrs)):
        raise ValueError("no offset produced a usable correlation")

    best = int(np.nanargmax(np.abs(corrs)))
    peak_corr = float(corrs[best])
    peak_lag_ms = int(lags[best] * grid_ms)

    zero_i = int(np.where(lags == 0)[0][0])
    zero_corr = float(corrs[zero_i]) if np.isfinite(corrs[zero_i]) else float("nan")

    # Fisher z on the peak correlation. Šidák across every offset scanned,
    # because taking the maximum over many offsets is itself a search.
    n = len(rb) - abs(int(lags[best]))
    n_tested = int(np.isfinite(corrs).sum())
    alpha_adj = 1 - (1 - alpha) ** (1 / max(1, n_tested))
    if n > 3:
        # atanh diverges at +/-1, which a synthetic exact copy hits. Clamping
        # keeps the z finite and enormous; bailing out to z=0 would report the
        # strongest possible evidence as no evidence at all.
        r = min(max(peak_corr, -1.0 + 1e-15), 1.0 - 1e-15)
        z = math.atanh(r) * math.sqrt(n - 3)
        p = float(2 * stats.norm.sf(abs(z)))
    else:
        z, p = 0.0, 1.0

    return LagEstimate(
        peak_lag_ms=peak_lag_ms,
        peak_corr=peak_corr,
        zero_lag_corr=zero_corr,
        n_samples=len(rb),
        grid_ms=grid_ms,
        lags_ms=lags * grid_ms,
        corrs=corrs,
        z_score=z,
        p_value=p,
        alpha_adjusted=alpha_adj,
        significant=bool(p < alpha_adj),
        notes=notes,
    )
