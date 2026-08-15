"""Event-driven backtester.

Vectorised backtests are faster and wrong for this problem. They make it easy to
compute a signal from a whole array and settle it against the same array, which
quietly leaks the future into the past. This engine instead walks the series tick
by tick and hands the strategy a `MarketView` that *cannot* see past the current
index — indexing forward raises. Lookahead becomes a crash rather than a
suspiciously good Sharpe ratio.

The engine also enforces what a real bot faces: one position at a time by
default, entry latency between decision and strike, and a risk layer that can
halt trading mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..data.store import TickSeries
from ..risk.breakers import BreakerState, CircuitBreakers
from ..risk.sizing import SizingPolicy
from .contract import CALL, PUT, BinaryContract, Direction

__all__ = ["MarketView", "LookaheadError", "BacktestResult", "run_backtest", "CALL", "PUT"]


class LookaheadError(IndexError):
    """Raised when a strategy tries to read data it could not have had."""


class MarketView:
    """Read-only window over a price series, truncated at the current tick.

    `view.price` is now. `view.history(n)` is the last n prices inclusive.
    Anything at or beyond the next index raises `LookaheadError`.
    """

    __slots__ = ("_series", "_i")

    def __init__(self, series: TickSeries, i: int) -> None:
        self._series = series
        self._i = i

    @property
    def i(self) -> int:
        return self._i

    @property
    def ts(self) -> int:
        return int(self._series.ts[self._i])

    @property
    def price(self) -> float:
        return float(self._series.price[self._i])

    @property
    def symbol(self) -> str:
        return self._series.symbol

    def history(self, n: int) -> np.ndarray:
        """Last `n` prices ending at the current tick. Shorter near the start."""
        if n <= 0:
            raise ValueError("n must be positive")
        lo = max(0, self._i + 1 - n)
        return self._series.price[lo : self._i + 1]

    def ts_history(self, n: int) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be positive")
        lo = max(0, self._i + 1 - n)
        return self._series.ts[lo : self._i + 1]

    def at(self, offset: int) -> float:
        """Price `offset` ticks back (offset >= 0). Forward access is an error."""
        if offset < 0:
            raise LookaheadError(
                f"strategy requested offset {offset}: future data is not available"
            )
        j = self._i - offset
        if j < 0:
            raise IndexError(f"offset {offset} predates the start of the series")
        return float(self._series.price[j])


#: A strategy returns +1 (call), -1 (put), or None (no trade) for the current tick.
Strategy = Callable[[MarketView], Optional[Direction]]


@dataclass
class BacktestResult:
    blotter: pd.DataFrame
    equity: np.ndarray
    contract: BinaryContract
    starting_equity: float
    halted_reason: Optional[str] = None
    skipped_busy: int = 0
    skipped_blocked: int = 0
    skipped_truncated: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return len(self.blotter)

    @property
    def final_equity(self) -> float:
        return float(self.equity[-1]) if len(self.equity) else self.starting_equity

    def win_rate(self) -> float:
        """Wins over decided trades, excluding refunded ties."""
        if self.blotter.empty:
            return float("nan")
        decided = self.blotter[self.blotter["ret"] != 0.0]
        if decided.empty:
            return float("nan")
        return float(decided["win"].sum()) / len(decided)

    def max_drawdown(self) -> float:
        if len(self.equity) == 0:
            return 0.0
        peak = np.maximum.accumulate(self.equity)
        return float(np.max((peak - self.equity) / peak))

    def summary(self) -> dict:
        wr = self.win_rate()
        be = self.contract.break_even_rate
        return {
            "trades": self.n_trades,
            "win_rate": wr,
            "break_even_rate": be,
            "edge_pp": (wr - be) * 100 if wr == wr else float("nan"),
            "ev_per_trade": self.contract.expected_value(wr) if wr == wr else float("nan"),
            "final_equity": self.final_equity,
            "return_pct": (self.final_equity / self.starting_equity - 1) * 100,
            "max_drawdown_pct": self.max_drawdown() * 100,
            "halted_reason": self.halted_reason,
            "skipped_busy": self.skipped_busy,
            "skipped_blocked": self.skipped_blocked,
            "skipped_truncated": self.skipped_truncated,
        }


def run_backtest(
    series: TickSeries,
    strategy: Strategy,
    contract: BinaryContract,
    *,
    sizing: Optional[SizingPolicy] = None,
    breakers: Optional[CircuitBreakers] = None,
    starting_equity: float = 1000.0,
    allow_concurrent: bool = False,
    warmup: int = 0,
) -> BacktestResult:
    """Walk `series` tick by tick, trading `strategy` under `contract`.

    Args:
        allow_concurrent: If False (default), no new trade is opened while one is
            open. This matches how a simple bot behaves and, importantly, limits
            label overlap — heavily overlapping trades share outcome information
            and inflate apparent significance.
        warmup: Ticks to skip before trading, for indicator windows to fill.
    """
    if len(series) == 0:
        raise ValueError("cannot backtest an empty series")

    sizing = sizing or SizingPolicy()
    breakers = breakers or CircuitBreakers()
    bstate = BreakerState(equity=starting_equity, peak_equity=starting_equity)

    equity = starting_equity
    equity_curve: list[float] = []
    rows: list[dict] = []
    busy_until = -1
    skipped_busy = 0
    skipped_blocked = 0
    skipped_truncated = 0
    halted_reason: Optional[str] = None

    last_ts = int(series.ts[-1])
    n = len(series)

    for i in range(warmup, n):
        ts = int(series.ts[i])

        if not allow_concurrent and ts < busy_until:
            continue

        halt = breakers.check(bstate, ts)
        if halt is not None:
            if bstate.halted_reason:
                # Latched: ruin or max drawdown. These do not resolve, so the
                # run is over.
                halted_reason = halt
                break
            # Transient: a daily loss limit, loss streak, or trade cap. Trading
            # resumes when the UTC day rolls, so skip this tick rather than
            # ending the backtest — treating a day-scoped pause as terminal
            # silently truncates the sample and biases every statistic computed
            # from it.
            skipped_blocked += 1
            continue

        direction = strategy(MarketView(series, i))
        if direction is None:
            continue
        if direction not in (CALL, PUT):
            raise ValueError(f"strategy returned invalid direction {direction!r}")

        entry_ts = contract.entry_ts(ts)
        expiry_ts = contract.expiry_ts(entry_ts)
        if expiry_ts > last_ts:
            # Not enough data to settle honestly; stop rather than guess.
            skipped_truncated += 1
            break

        raw_strike = series.price_at(entry_ts)
        expiry_price = series.price_at(expiry_ts)
        if raw_strike is None or expiry_price is None:
            skipped_truncated += 1
            continue

        strike = contract.effective_strike(raw_strike, direction)
        ret = contract.settle(strike, expiry_price, direction)

        stake = sizing.stake(equity, bstate)
        if stake <= 0:
            skipped_busy += 1
            continue

        pnl = stake * ret
        equity += pnl
        equity_curve.append(equity)

        bstate.update(equity=equity, pnl=pnl, ret=ret, ts=ts)
        busy_until = expiry_ts

        rows.append(
            {
                "signal_ts": ts,
                "entry_ts": entry_ts,
                "expiry_ts": expiry_ts,
                "direction": int(direction),
                "strike": strike,
                "expiry_price": expiry_price,
                "stake": stake,
                "ret": ret,
                "pnl": pnl,
                "equity": equity,
                "win": ret > 0.0,
            }
        )

        if equity <= 0:
            halted_reason = "ruin: equity reached zero"
            break

    blotter = pd.DataFrame(
        rows,
        columns=[
            "signal_ts", "entry_ts", "expiry_ts", "direction", "strike",
            "expiry_price", "stake", "ret", "pnl", "equity", "win",
        ],
    )
    return BacktestResult(
        blotter=blotter,
        equity=np.array(equity_curve, dtype=np.float64),
        contract=contract,
        starting_equity=starting_equity,
        halted_reason=halted_reason,
        skipped_busy=skipped_busy,
        skipped_blocked=skipped_blocked,
        skipped_truncated=skipped_truncated,
    )
