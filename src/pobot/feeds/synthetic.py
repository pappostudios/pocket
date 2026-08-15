"""Deterministic synthetic feeds.

Two jobs:

1. Make the whole downstream pipeline testable without a network connection or a
   broker account. Every component below the feed layer is exercised by these.

2. Provide *known-answer* price processes for validating the validator. A
   driftless random walk has no exploitable structure, so any strategy that
   shows a significant edge on `RandomWalkFeed` has revealed a bug in the
   backtester or the statistics — not a discovery. This is the single most
   useful test in the repo: it catches lookahead bias, which is the reason most
   retail bots backtest profitably and lose money live.

`MeanRevertingFeed` is the counterpart: a process with genuine, tunable
structure. A correct pipeline must find an edge there. If it cannot detect a
signal you deliberately planted, it will not detect a real one either.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np

from .base import FeedAdapter, Millis, Tick


class RandomWalkFeed(FeedAdapter):
    """Driftless geometric random walk. Contains no exploitable structure.

    Use as the null hypothesis. Expected out-of-sample win rate of *any*
    strategy on this feed is exactly 50%, and therefore expected value at any
    realistic payout is firmly negative.
    """

    name = "synthetic_rw"

    def __init__(
        self,
        symbols: list[str],
        *,
        start_price: float = 1.1000,
        vol_per_tick: float = 1e-4,
        interval_ms: int = 100,
        seed: int = 0,
        start_ts: Millis = 1_700_000_000_000,
    ) -> None:
        super().__init__(symbols)
        self.start_price = start_price
        self.vol_per_tick = vol_per_tick
        self.interval_ms = interval_ms
        self.seed = seed
        self.start_ts = start_ts

    def generate(self, n: int) -> list[Tick]:
        """Produce `n` ticks per symbol synchronously (no event loop needed)."""
        rng = np.random.default_rng(self.seed)
        out: list[Tick] = []
        for sym_i, symbol in enumerate(self.symbols):
            # Decorrelate symbols so multi-symbol tests aren't accidentally
            # trading the same series under two names.
            sym_rng = np.random.default_rng(self.seed + 1000 * (sym_i + 1))
            shocks = sym_rng.normal(0.0, self.vol_per_tick, size=n)
            prices = self.start_price * np.exp(np.cumsum(shocks))
            for i in range(n):
                ts = self.start_ts + i * self.interval_ms
                out.append(
                    Tick(
                        symbol=symbol,
                        price=float(prices[i]),
                        ts_event=ts,
                        ts_recv=ts,
                        source=self.name,
                    )
                )
        del rng
        out.sort(key=lambda t: (t.ts_recv, t.symbol))
        return out

    async def stream(self) -> AsyncIterator[Tick]:
        i = 0
        rng = np.random.default_rng(self.seed)
        price = {s: self.start_price for s in self.symbols}
        while True:
            ts = self.start_ts + i * self.interval_ms
            for symbol in self.symbols:
                price[symbol] *= float(np.exp(rng.normal(0.0, self.vol_per_tick)))
                tick = Tick(
                    symbol=symbol,
                    price=price[symbol],
                    ts_event=ts,
                    ts_recv=ts,
                    source=self.name,
                )
                self.stats.record(tick)
                yield tick
            i += 1
            await asyncio.sleep(0)


class MeanRevertingFeed(FeedAdapter):
    """Ornstein-Uhlenbeck process — a deliberately planted, detectable edge.

    Models the hypothesis worth testing about broker-generated "OTC" weekend
    assets: that they are synthetic series with statistical structure a real
    market would arbitrage away. Whether any *actual* platform's generator looks
    like this is an empirical question your captured data answers. This class
    exists so you can confirm the pipeline *would* notice if it did.

    Higher `kappa` means faster reversion and a stronger, easier-to-detect edge.
    """

    name = "synthetic_ou"

    def __init__(
        self,
        symbols: list[str],
        *,
        mean: float = 1.1000,
        kappa: float = 0.05,
        vol_per_tick: float = 1e-4,
        interval_ms: int = 100,
        seed: int = 0,
        start_ts: Millis = 1_700_000_000_000,
    ) -> None:
        super().__init__(symbols)
        self.mean = mean
        self.kappa = kappa
        self.vol_per_tick = vol_per_tick
        self.interval_ms = interval_ms
        self.seed = seed
        self.start_ts = start_ts

    def generate(self, n: int) -> list[Tick]:
        out: list[Tick] = []
        for sym_i, symbol in enumerate(self.symbols):
            rng = np.random.default_rng(self.seed + 1000 * (sym_i + 1))
            price = self.mean
            for i in range(n):
                # dX = kappa * (mu - X) dt + sigma dW, unit dt per tick.
                price += self.kappa * (self.mean - price) + rng.normal(
                    0.0, self.vol_per_tick
                )
                ts = self.start_ts + i * self.interval_ms
                out.append(
                    Tick(
                        symbol=symbol,
                        price=float(price),
                        ts_event=ts,
                        ts_recv=ts,
                        source=self.name,
                    )
                )
        out.sort(key=lambda t: (t.ts_recv, t.symbol))
        return out

    async def stream(self) -> AsyncIterator[Tick]:
        rng = np.random.default_rng(self.seed)
        price = {s: self.mean for s in self.symbols}
        i = 0
        while True:
            ts = self.start_ts + i * self.interval_ms
            for symbol in self.symbols:
                price[symbol] += self.kappa * (self.mean - price[symbol]) + rng.normal(
                    0.0, self.vol_per_tick
                )
                tick = Tick(
                    symbol=symbol,
                    price=price[symbol],
                    ts_event=ts,
                    ts_recv=ts,
                    source=self.name,
                )
                self.stats.record(tick)
                yield tick
            i += 1
            await asyncio.sleep(0)


class LaggedFeed(FeedAdapter):
    """Wraps another feed and republishes its prices `lag_ms` later.

    This is the latency-arbitrage hypothesis in executable form: if the broker's
    quote stream is a delayed copy of the real market, then the reference feed's
    *current* price predicts the broker's *next* price, and the edge is
    mechanical rather than statistical.

    Use it to calibrate detection power before pointing the analysis at real
    captured data — it answers "how many hours of dual-feed capture do I need to
    detect a 200ms lag?" without spending the hours first.
    """

    name = "synthetic_lagged"

    def __init__(self, inner: RandomWalkFeed | MeanRevertingFeed, lag_ms: int) -> None:
        super().__init__(inner.symbols)
        self.inner = inner
        self.lag_ms = lag_ms

    def generate(self, n: int) -> list[Tick]:
        base = self.inner.generate(n)
        return [
            Tick(
                symbol=t.symbol,
                price=t.price,
                ts_event=t.ts_event + self.lag_ms,
                ts_recv=t.ts_recv + self.lag_ms,
                source=self.name,
                bid=t.bid,
                ask=t.ask,
            )
            for t in base
        ]

    async def stream(self) -> AsyncIterator[Tick]:
        async for tick in self.inner.stream():
            lagged = Tick(
                symbol=tick.symbol,
                price=tick.price,
                ts_event=tick.ts_event + self.lag_ms,
                ts_recv=tick.ts_recv + self.lag_ms,
                source=self.name,
            )
            self.stats.record(lagged)
            yield lagged
