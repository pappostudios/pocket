"""Feed abstraction.

Every price source — the broker's own stream, an independent reference feed, a
synthetic generator — produces the same `Tick` record so downstream code never
knows or cares where a price came from.

Two timestamps per tick, and the distinction is the whole point of this module:

  ts_event  the timestamp the source claims for the price
  ts_recv   the timestamp we observed it locally (monotonic-anchored wall clock)

`ts_event` is what the broker says. `ts_recv` is what actually happened. The gap
between them, measured against an independent feed carrying the same instrument,
is the only direct evidence you can gather about whether the broker's quote
stream lags the real market. Recording only one of the two throws that away
permanently, and no amount of later analysis recovers it.
"""

from __future__ import annotations

import abc
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Optional

# Every timestamp in this codebase is integer milliseconds since the Unix epoch,
# UTC. No naive datetimes, no floats-as-seconds, no local time. Mixing time
# representations is the most common source of silent lookahead bias in a
# backtest, so there is exactly one representation.
Millis = int


def now_ms() -> Millis:
    """Current wall-clock time in epoch milliseconds."""
    return int(time.time() * 1000)


@dataclass(frozen=True, slots=True)
class Tick:
    """A single observed price.

    `bid`/`ask` are optional because most retail binary-options streams publish
    only a single mid-like price. When they are absent the backtester cannot
    model spread cost, which it will tell you about rather than assume zero.
    """

    symbol: str
    price: float
    ts_event: Millis
    ts_recv: Millis
    source: str
    bid: Optional[float] = None
    ask: Optional[float] = None

    @property
    def latency_ms(self) -> int:
        """Observed delivery lag: how stale the price was when we saw it.

        Includes network transit *and* any clock offset between us and the
        source, so treat the absolute value as unreliable and the *changes* in
        it as the signal. A source whose latency drifts upward under load, or
        jumps when the real market moves, is telling you something.
        """
        return self.ts_recv - self.ts_event


@dataclass
class FeedStats:
    """Health counters for a running feed.

    Capture runs for weeks unattended. Without these you discover a feed died
    on day 3 only when the backtest produces nonsense in week 4.
    """

    ticks: int = 0
    reconnects: int = 0
    errors: int = 0
    last_tick_ms: Optional[Millis] = None
    symbols_seen: set[str] = field(default_factory=set)

    def record(self, tick: Tick) -> None:
        self.ticks += 1
        self.last_tick_ms = tick.ts_recv
        self.symbols_seen.add(tick.symbol)

    def staleness_ms(self, ref_ms: Optional[Millis] = None) -> Optional[int]:
        """Milliseconds since the last tick, or None if nothing has arrived."""
        if self.last_tick_ms is None:
            return None
        return (ref_ms if ref_ms is not None else now_ms()) - self.last_tick_ms


class FeedAdapter(abc.ABC):
    """Base class for a price source.

    Implementations must be *pure transport*: connect, decode, yield. No
    filtering, no resampling, no gap-filling, no dropping of ticks that look
    wrong. Capture records what the source actually sent; cleaning decisions
    belong in analysis, where they are visible and reversible. A cleaning rule
    baked into capture silently rewrites history and cannot be undone.
    """

    #: Short identifier written into every tick's `source` field.
    name: str = "unknown"

    def __init__(self, symbols: list[str]) -> None:
        self.symbols = list(symbols)
        self.stats = FeedStats()

    @abc.abstractmethod
    def stream(self) -> AsyncIterator[Tick]:
        """Yield ticks until cancelled.

        Implementations should reconnect internally on transport failure and
        increment `stats.reconnects`, rather than terminating the iterator — the
        recorder treats iterator exhaustion as a permanent failure.
        """
        raise NotImplementedError
