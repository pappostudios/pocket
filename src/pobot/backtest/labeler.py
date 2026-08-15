"""Turn signal timestamps into settled binary outcomes.

Labelling is where backtests are usually broken, and the breakage is always the
same shape: the label is computed from information the strategy could not have
had. Three rules keep it honest here.

1. Price at time t is the last tick *at or before* t. Never the next tick, never
   an interpolation between them. At time t you know the most recent print.

2. The strike is resolved at `signal_ts + entry_latency_ms`, not at `signal_ts`.
   The bot decides, then time passes, then the strike is fixed at whatever the
   price has become.

3. A contract whose expiry falls beyond the end of the data is *dropped*, not
   settled at the last known price. Silently settling truncated contracts biases
   the final trades toward whatever the series happened to be doing when capture
   stopped.

Each label also carries `t1` — its expiry time. The purged cross-validator needs
it to know which training samples overlap a test window, and dropping it is what
makes overlapping-label leakage invisible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.store import TickSeries
from .contract import BinaryContract, Direction


@dataclass
class LabelSet:
    """Settled trades, aligned row-for-row.

    Attributes:
        signal_ts: When the strategy decided.
        entry_ts:  When the strike was fixed (t0).
        expiry_ts: When the contract settled (t1) — required for purged CV.
        direction: +1 call, -1 put.
        strike / expiry_price: Prices used for settlement.
        ret: Net return per unit stake (+payout, 0 on refunded tie, -1).
        win: Boolean win flag. Refunded ties are False here but excluded from
            win-rate denominators by `win_rate()`, since a tie is neither a win
            nor a loss and counting it as a loss understates a real edge.
    """

    signal_ts: np.ndarray
    entry_ts: np.ndarray
    expiry_ts: np.ndarray
    direction: np.ndarray
    strike: np.ndarray
    expiry_price: np.ndarray
    ret: np.ndarray
    win: np.ndarray

    def __len__(self) -> int:
        return len(self.signal_ts)

    @property
    def ties(self) -> np.ndarray:
        return self.ret == 0.0

    def win_rate(self) -> float:
        """Wins / (wins + losses), excluding refunded ties."""
        decided = ~self.ties
        n = int(decided.sum())
        if n == 0:
            return float("nan")
        return float(self.win[decided].sum()) / n

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "signal_ts": self.signal_ts,
                "entry_ts": self.entry_ts,
                "expiry_ts": self.expiry_ts,
                "direction": self.direction,
                "strike": self.strike,
                "expiry_price": self.expiry_price,
                "ret": self.ret,
                "win": self.win,
            }
        )

    def subset(self, idx: np.ndarray) -> "LabelSet":
        return LabelSet(
            signal_ts=self.signal_ts[idx],
            entry_ts=self.entry_ts[idx],
            expiry_ts=self.expiry_ts[idx],
            direction=self.direction[idx],
            strike=self.strike[idx],
            expiry_price=self.expiry_price[idx],
            ret=self.ret[idx],
            win=self.win[idx],
        )


def _price_at_vec(series: TickSeries, ts: np.ndarray) -> np.ndarray:
    """Vectorised last-price-at-or-before. NaN where the series starts later."""
    idx = np.searchsorted(series.ts, ts, side="right") - 1
    out = np.full(len(ts), np.nan, dtype=np.float64)
    valid = idx >= 0
    out[valid] = series.price[idx[valid]]
    return out


def label(
    series: TickSeries,
    signal_ts: np.ndarray,
    directions: np.ndarray,
    contract: BinaryContract,
) -> LabelSet:
    """Settle a batch of signals against a price series.

    Contracts expiring past the end of `series` are dropped, as are any whose
    strike or expiry price cannot be resolved.
    """
    signal_ts = np.asarray(signal_ts, dtype=np.int64)
    directions = np.asarray(directions, dtype=np.int64)
    if len(signal_ts) != len(directions):
        raise ValueError("signal_ts and directions must be the same length")
    if len(series) == 0:
        raise ValueError("cannot label against an empty price series")
    if not np.all(np.isin(directions, (-1, 1))):
        raise ValueError("directions must contain only +1 or -1")

    entry_ts = signal_ts + contract.entry_latency_ms
    if contract.expiry_mode == "relative":
        expiry_ts = entry_ts + contract.duration_s * 1000
    else:
        dur_ms = contract.duration_s * 1000
        expiry_ts = ((entry_ts // dur_ms) + 1) * dur_ms

    raw_strike = _price_at_vec(series, entry_ts)
    expiry_price = _price_at_vec(series, expiry_ts)

    # Drop anything the data cannot support rather than guessing a fill.
    last_ts = int(series.ts[-1])
    ok = (
        ~np.isnan(raw_strike)
        & ~np.isnan(expiry_price)
        & (expiry_ts <= last_ts)
    )

    signal_ts, entry_ts, expiry_ts = signal_ts[ok], entry_ts[ok], expiry_ts[ok]
    directions, raw_strike, expiry_price = directions[ok], raw_strike[ok], expiry_price[ok]

    strike = raw_strike + directions * contract.spread

    ret = np.where(
        expiry_price == strike,
        0.0 if contract.tie_rule == "refund" else -1.0,
        np.where(
            (expiry_price > strike) == (directions == 1),
            contract.payout,
            -1.0,
        ),
    ).astype(np.float64)

    win = ret > 0.0

    return LabelSet(
        signal_ts=signal_ts,
        entry_ts=entry_ts,
        expiry_ts=expiry_ts,
        direction=directions,
        strike=strike,
        expiry_price=expiry_price,
        ret=ret,
        win=win,
    )
