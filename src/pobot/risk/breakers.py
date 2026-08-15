"""Circuit breakers.

Sizing decides how much to risk when things go normally. Breakers decide when to
stop, and they matter more, because the situation they guard against is the one
where your win-rate estimate was simply wrong.

Every limit is checked *before* a trade opens, not after. A breaker that trips
only on review is a report, not a control.

The defaults are intentionally tight. Loosening them is a decision to make
deliberately, with a reason, not a knob to turn because the backtest halted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _utc_day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


@dataclass
class BreakerState:
    """Mutable trading state the breakers read."""

    equity: float
    peak_equity: float
    consecutive_losses: int = 0
    trades_today: int = 0
    day_key: Optional[str] = None
    day_start_equity: Optional[float] = None
    last_ts: Optional[int] = None
    halted_reason: Optional[str] = None
    history: list[float] = field(default_factory=list)

    def roll_day(self, ts_ms: int) -> None:
        """Reset per-day counters when the UTC date changes.

        The loss streak resets here too. Without it a streak breaker is
        permanent rather than day-scoped: once trading is blocked no further
        trades occur, so no win can ever arrive to clear the counter.
        """
        key = _utc_day(ts_ms)
        if self.day_key != key:
            self.day_key = key
            self.day_start_equity = self.equity
            self.trades_today = 0
            self.consecutive_losses = 0

    def update(self, *, equity: float, pnl: float, ret: float, ts: int) -> None:
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        self.trades_today += 1
        self.last_ts = ts
        self.history.append(ret)
        if ret < 0:
            self.consecutive_losses += 1
        elif ret > 0:
            self.consecutive_losses = 0
        # A refunded tie (ret == 0) leaves the streak untouched: no capital was
        # lost, so it is not evidence the strategy is failing.

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    def daily_loss(self) -> float:
        if self.day_start_equity is None or self.day_start_equity <= 0:
            return 0.0
        return max(0.0, (self.day_start_equity - self.equity) / self.day_start_equity)


@dataclass
class CircuitBreakers:
    """Pre-trade limits. `check()` returns a halt reason, or None to proceed.

    Attributes:
        max_daily_loss: Halt for the rest of the UTC day after losing this
            fraction of the day's opening equity.
        max_drawdown: Halt permanently at this drawdown from peak equity. This
            is the "the edge was not real" breaker — reaching it means the live
            result has diverged far enough from the backtest that continuing is
            no longer supported by evidence.
        max_consecutive_losses: Halt after this many losses in a row. Not a
            statistical necessity so much as a tripwire for a broken feed, a
            stale model, or a payout change nobody noticed.
        max_trades_per_day: Caps exposure to per-trade house edge. At a -4%
            edge, trade count is the multiplier on expected loss.
        max_feed_staleness_ms: Refuse to trade on prices older than this. Trading
            a frozen feed is how a bot places a hundred orders into a market that
            has already moved.
    """

    max_daily_loss: float = 0.10
    max_drawdown: float = 0.25
    max_consecutive_losses: int = 6
    max_trades_per_day: int = 200
    max_feed_staleness_ms: int = 5_000

    def check(
        self, state: BreakerState, ts_ms: int, *, last_tick_ms: Optional[int] = None
    ) -> Optional[str]:
        if state.halted_reason:
            return state.halted_reason

        state.roll_day(ts_ms)

        if state.equity <= 0:
            return self._halt(state, "equity exhausted")

        if state.drawdown >= self.max_drawdown:
            return self._halt(
                state,
                f"max drawdown breached: {state.drawdown:.1%} >= {self.max_drawdown:.1%}",
            )

        if last_tick_ms is not None and (ts_ms - last_tick_ms) > self.max_feed_staleness_ms:
            # Transient by nature, so it blocks this trade without latching.
            return f"feed stale by {ts_ms - last_tick_ms}ms"

        # Day-scoped limits below: they block trading until the UTC date rolls,
        # and deliberately do not latch into halted_reason.
        if state.daily_loss() >= self.max_daily_loss:
            return (
                f"daily loss limit: {state.daily_loss():.1%} >= {self.max_daily_loss:.1%}"
            )

        if state.consecutive_losses >= self.max_consecutive_losses:
            return f"consecutive losses: {state.consecutive_losses}"

        if state.trades_today >= self.max_trades_per_day:
            return f"daily trade cap: {state.trades_today}"

        return None

    @staticmethod
    def _halt(state: BreakerState, reason: str) -> str:
        """Latch a permanent halt — these conditions do not resolve on their own."""
        state.halted_reason = reason
        return reason
