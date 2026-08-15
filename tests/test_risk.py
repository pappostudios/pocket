import pytest

from pobot.risk.breakers import BreakerState, CircuitBreakers
from pobot.risk.sizing import MartingaleSizing, SizingPolicy, kelly_fraction

DAY = 86_400_000


def test_kelly_is_zero_at_break_even():
    assert kelly_fraction(1 / 1.92, 0.92) == pytest.approx(0.0, abs=1e-12)


def test_kelly_at_55_percent_is_about_6_percent():
    assert kelly_fraction(0.55, 0.92) == pytest.approx(0.0609, abs=1e-3)


def test_kelly_negative_below_break_even():
    assert kelly_fraction(0.50, 0.92) < 0


def test_losing_strategies_are_sized_to_zero():
    assert SizingPolicy(win_rate=0.50, payout=0.92).stake(1000.0) == 0.0


def test_fractional_kelly_is_a_quarter_of_full():
    full = kelly_fraction(0.60, 0.92)
    assert SizingPolicy(win_rate=0.60, payout=0.92, fraction=0.25,
                        max_fraction=1.0).target_fraction() == pytest.approx(full * 0.25)


def test_max_fraction_caps_an_aggressive_edge():
    p = SizingPolicy(win_rate=0.90, payout=0.92, fraction=1.0, max_fraction=0.02)
    assert p.target_fraction() == 0.02


def test_stake_below_broker_minimum_trades_nothing():
    """Rounding up to the minimum would silently oversize the bet."""
    p = SizingPolicy(win_rate=0.60, payout=0.92, min_stake=10.0)
    assert p.stake(100.0) == 0.0


def test_martingale_refuses_with_an_explanation():
    with pytest.raises(NotImplementedError, match="2.17"):
        MartingaleSizing()


# --- Circuit breakers -------------------------------------------------------

def fresh(equity=1000.0):
    return BreakerState(equity=equity, peak_equity=equity)


def test_healthy_state_permits_trading():
    assert CircuitBreakers().check(fresh(), DAY) is None


def test_drawdown_breaker_latches_permanently():
    s = fresh()
    s.equity = 700.0  # 30% below peak
    b = CircuitBreakers(max_drawdown=0.25)
    assert b.check(s, DAY) is not None
    s.equity = 1000.0  # recovery does not clear a latched halt
    assert b.check(s, DAY) is not None


def test_daily_loss_breaker_blocks_then_clears_next_day():
    s = fresh()
    b = CircuitBreakers(max_daily_loss=0.10, max_drawdown=0.99)
    assert b.check(s, DAY) is None
    s.equity = 880.0
    assert "daily loss" in (b.check(s, DAY) or "")
    assert b.check(s, DAY * 2) is None  # new UTC day resets the baseline


def test_consecutive_loss_breaker():
    s = fresh()
    b = CircuitBreakers(max_consecutive_losses=3)
    b.check(s, DAY)  # establish the trading day before counting losses into it
    for _ in range(3):
        s.update(equity=s.equity - 10, pnl=-10, ret=-1.0, ts=DAY)
    assert "consecutive" in (b.check(s, DAY) or "")


def test_loss_streak_clears_on_the_next_day():
    """Day-scoped, like the daily loss limit — otherwise a blocked bot can never
    place the winning trade that would reset the counter."""
    s = fresh()
    b = CircuitBreakers(max_consecutive_losses=3)
    b.check(s, DAY)
    for _ in range(3):
        s.update(equity=s.equity - 10, pnl=-10, ret=-1.0, ts=DAY)
    assert b.check(s, DAY) is not None
    assert b.check(s, DAY * 2) is None


def test_a_win_resets_the_loss_streak():
    s = fresh()
    for _ in range(3):
        s.update(equity=s.equity - 10, pnl=-10, ret=-1.0, ts=DAY)
    s.update(equity=s.equity + 9.2, pnl=9.2, ret=0.92, ts=DAY)
    assert s.consecutive_losses == 0


def test_refunded_tie_leaves_the_streak_untouched():
    s = fresh()
    s.update(equity=s.equity - 10, pnl=-10, ret=-1.0, ts=DAY)
    s.update(equity=s.equity, pnl=0.0, ret=0.0, ts=DAY)
    assert s.consecutive_losses == 1


def test_daily_trade_cap():
    s = fresh()
    b = CircuitBreakers(max_trades_per_day=5, max_drawdown=0.99, max_daily_loss=0.99)
    b.check(s, DAY)
    for _ in range(5):
        s.update(equity=s.equity + 1, pnl=1, ret=0.92, ts=DAY)
    assert "trade cap" in (b.check(s, DAY) or "")


def test_stale_feed_blocks_trading():
    b = CircuitBreakers(max_feed_staleness_ms=5000)
    assert "stale" in (b.check(fresh(), DAY, last_tick_ms=DAY - 10_000) or "")


def test_fresh_feed_permits_trading():
    b = CircuitBreakers(max_feed_staleness_ms=5000)
    assert b.check(fresh(), DAY, last_tick_ms=DAY - 100) is None


def test_zero_equity_halts():
    s = fresh(0.0)
    assert CircuitBreakers().check(s, DAY) is not None
