"""Engine tests, including the one that matters most.

`test_no_edge_on_random_walk` is the load-bearing test in this repository. A
driftless random walk contains no exploitable structure, so any strategy must
land at ~50% and lose money at any realistic payout. If this test ever reports a
profitable strategy, the pipeline has a lookahead bug and every other result is
worthless.
"""

import numpy as np
import pytest

from pobot.backtest.contract import CALL, PUT, BinaryContract
from pobot.backtest.engine import LookaheadError, MarketView, run_backtest
from pobot.data.store import TickSeries, ticks_to_series
from pobot.feeds.synthetic import MeanRevertingFeed, RandomWalkFeed
from pobot.risk.breakers import CircuitBreakers
from pobot.risk.sizing import SizingPolicy

#: Risk limits exist to protect capital, not to shape research results. Letting
#: the default breakers halt an edge-measurement run truncates the sample at the
#: worst possible moment — right after a loss streak — which biases the measured
#: win rate upward. Measurement runs disable them; live sizing never would.
NO_LIMITS = CircuitBreakers(
    max_daily_loss=1.0,
    max_drawdown=1.0,
    max_consecutive_losses=10**9,
    max_trades_per_day=10**9,
)


def const_call(view):
    return CALL


def make_series(prices, step_ms=1000, start=1_000_000):
    return TickSeries(
        symbol="X", source="test",
        ts=np.arange(start, start + len(prices) * step_ms, step_ms, dtype=np.int64),
        price=np.array(prices, dtype=np.float64),
    )


# --- MarketView: lookahead must be structurally impossible ------------------

def test_market_view_exposes_only_the_present():
    s = make_series([1.0, 2.0, 3.0, 4.0])
    v = MarketView(s, 2)
    assert v.price == 3.0
    assert v.at(0) == 3.0
    assert v.at(2) == 1.0
    assert list(v.history(2)) == [2.0, 3.0]


def test_market_view_rejects_future_access():
    v = MarketView(make_series([1.0, 2.0, 3.0]), 1)
    with pytest.raises(LookaheadError):
        v.at(-1)


def test_history_truncates_at_series_start():
    v = MarketView(make_series([1.0, 2.0, 3.0]), 1)
    assert list(v.history(10)) == [1.0, 2.0]


# --- Core no-edge guarantee -------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_no_edge_on_random_walk(seed):
    """No strategy can beat break-even on a driftless random walk."""
    feed = RandomWalkFeed(["X"], interval_ms=1000, seed=seed)
    series = ticks_to_series(feed.generate(6000), "X", feed.name)
    contract = BinaryContract(payout=0.92, duration_s=60, entry_latency_ms=250)

    def momentum(view):
        h = view.history(10)
        return None if len(h) < 10 else (CALL if h[-1] > h[0] else PUT)

    res = run_backtest(series, momentum, contract,
                       sizing=SizingPolicy(flat_stake=0.01), breakers=NO_LIMITS,
                       warmup=20)

    assert res.n_trades > 20
    # Sampling noise is wide at this n; the point is that it sits around a coin
    # flip rather than anywhere near a genuine edge.
    assert 0.30 < res.win_rate() < 0.70


def test_planted_edge_is_detected():
    """A strongly mean-reverting series must beat break-even. If this fails,
    the pipeline cannot find a real signal either."""
    feed = MeanRevertingFeed(["X"], kappa=0.30, interval_ms=1000, seed=11)
    series = ticks_to_series(feed.generate(6000), "X", feed.name)
    contract = BinaryContract(payout=0.92, duration_s=60, entry_latency_ms=250)

    def reversion(view):
        h = view.history(20)
        return None if len(h) < 20 else (PUT if h[-1] > h.mean() else CALL)

    res = run_backtest(series, reversion, contract,
                       sizing=SizingPolicy(flat_stake=0.01), breakers=NO_LIMITS,
                       warmup=30)
    assert res.win_rate() > contract.break_even_rate


# --- Execution realism ------------------------------------------------------

def test_no_concurrent_positions_by_default():
    series = make_series([1.0 + 0.001 * i for i in range(400)])
    contract = BinaryContract(payout=0.9, duration_s=60, entry_latency_ms=0)
    res = run_backtest(series, const_call, contract, sizing=SizingPolicy(flat_stake=0.01))
    b = res.blotter
    # Each trade must open at or after the previous one settled.
    assert (b["signal_ts"].to_numpy()[1:] >= b["expiry_ts"].to_numpy()[:-1]).all()


def test_concurrent_positions_when_enabled():
    series = make_series([1.0 + 0.001 * i for i in range(400)])
    contract = BinaryContract(payout=0.9, duration_s=60, entry_latency_ms=0)
    serial = run_backtest(series, const_call, contract, sizing=SizingPolicy(flat_stake=0.01))
    overlapped = run_backtest(series, const_call, contract,
                              sizing=SizingPolicy(flat_stake=0.01), allow_concurrent=True)
    assert overlapped.n_trades > serial.n_trades


def test_day_scoped_breaker_pauses_without_ending_the_run():
    """A loss streak blocks trading for the day; it must not end the backtest.

    Treating a day-scoped pause as terminal truncates the sample immediately
    after a losing run, which biases the measured win rate upward.
    """
    series = make_series([1.0 - 0.001 * i for i in range(600)])
    contract = BinaryContract(payout=0.9, duration_s=10, entry_latency_ms=0)
    res = run_backtest(
        series, const_call, contract, sizing=SizingPolicy(flat_stake=0.01),
        breakers=CircuitBreakers(max_consecutive_losses=3, max_drawdown=1.0,
                                 max_daily_loss=1.0, max_trades_per_day=10**9),
    )
    assert res.halted_reason is None
    assert res.skipped_blocked > 0


def test_latched_breaker_ends_the_run():
    """Max drawdown does not resolve on its own, so it stops the run for good."""
    series = make_series([1.0 - 0.001 * i for i in range(600)])
    contract = BinaryContract(payout=0.9, duration_s=10, entry_latency_ms=0)
    res = run_backtest(
        series, const_call, contract, sizing=SizingPolicy(flat_stake=0.02),
        breakers=CircuitBreakers(max_drawdown=0.05, max_daily_loss=1.0,
                                 max_consecutive_losses=10**9,
                                 max_trades_per_day=10**9),
    )
    assert res.halted_reason is not None
    assert "drawdown" in res.halted_reason


def test_truncated_contracts_are_never_settled():
    """The engine stops rather than settling at the last known price."""
    series = make_series([1.0] * 100)  # 100s of data
    contract = BinaryContract(payout=0.9, duration_s=60, entry_latency_ms=0)
    res = run_backtest(series, const_call, contract, sizing=SizingPolicy(flat_stake=0.01))
    last_ts = int(series.ts[-1])
    assert (res.blotter["expiry_ts"] <= last_ts).all()


def test_rising_market_pays_calls():
    series = make_series([1.0 + 0.001 * i for i in range(400)])
    contract = BinaryContract(payout=0.9, duration_s=60, entry_latency_ms=0)
    res = run_backtest(series, const_call, contract, sizing=SizingPolicy(flat_stake=0.01))
    assert res.win_rate() == 1.0
    assert res.final_equity > res.starting_equity


def test_invalid_direction_from_strategy_is_rejected():
    series = make_series([1.0] * 300)
    with pytest.raises(ValueError):
        run_backtest(series, lambda v: 2, BinaryContract(duration_s=10),
                     sizing=SizingPolicy(flat_stake=0.01))


def test_empty_series_rejected():
    empty = TickSeries("X", "t", np.array([], dtype=np.int64), np.array([]))
    with pytest.raises(ValueError):
        run_backtest(empty, const_call, BinaryContract())


def test_summary_reports_edge_against_break_even():
    series = make_series([1.0 + 0.001 * i for i in range(400)])
    contract = BinaryContract(payout=0.92, duration_s=60, entry_latency_ms=0)
    res = run_backtest(series, const_call, contract, sizing=SizingPolicy(flat_stake=0.01))
    s = res.summary()
    assert s["break_even_rate"] == pytest.approx(1 / 1.92)
    assert s["edge_pp"] > 0
