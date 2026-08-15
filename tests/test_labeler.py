import numpy as np
import pytest

from pobot.backtest.contract import CALL, PUT, BinaryContract
from pobot.backtest.labeler import label
from pobot.data.store import TickSeries


def make_series(prices, step_ms=1000, start=1_000_000):
    return TickSeries(
        symbol="X",
        source="test",
        ts=np.arange(start, start + len(prices) * step_ms, step_ms, dtype=np.int64),
        price=np.array(prices, dtype=np.float64),
    )


def test_price_at_uses_last_tick_at_or_before():
    s = make_series([1.0, 2.0, 3.0])
    assert s.price_at(1_000_000) == 1.0
    assert s.price_at(1_000_500) == 1.0   # between ticks: the earlier one
    assert s.price_at(1_001_000) == 2.0
    assert s.price_at(1_009_999) == 3.0


def test_price_at_before_series_start_is_none():
    assert make_series([1.0, 2.0]).price_at(999_999) is None


def test_strike_is_taken_after_entry_latency():
    """Strike must reflect the price when the order lands, not when it was decided."""
    s = make_series([1.0, 5.0, 5.0, 5.0, 5.0])
    c = BinaryContract(payout=0.9, duration_s=2, entry_latency_ms=1000,
                       expiry_mode="relative")
    ls = label(s, np.array([1_000_000]), np.array([CALL]), c)
    assert len(ls) == 1
    assert ls.strike[0] == 5.0  # price at t+1000ms, not the 1.0 at signal time


def test_contracts_expiring_past_the_data_are_dropped():
    s = make_series([1.0] * 5)  # ends at 1_004_000
    c = BinaryContract(duration_s=60, entry_latency_ms=0)
    ls = label(s, np.array([1_000_000, 1_001_000]), np.array([CALL, PUT]), c)
    assert len(ls) == 0


def test_win_and_loss_are_settled_correctly():
    s = make_series([1.0, 1.0, 2.0, 2.0, 2.0])
    c = BinaryContract(payout=0.92, duration_s=2, entry_latency_ms=0,
                       expiry_mode="relative")
    ls = label(s, np.array([1_000_000, 1_000_000]), np.array([CALL, PUT]), c)
    assert ls.ret[0] == 0.92   # price rose, call wins
    assert ls.ret[1] == -1.0   # price rose, put loses


def test_ties_are_refunded_and_excluded_from_win_rate():
    s = make_series([1.0] * 10)
    c = BinaryContract(payout=0.92, duration_s=2, entry_latency_ms=0, tie_rule="refund")
    ls = label(s, np.array([1_000_000, 1_001_000]), np.array([CALL, CALL]), c)
    assert np.all(ls.ret == 0.0)
    assert ls.ties.all()
    assert np.isnan(ls.win_rate())  # no decided trades


def test_expiry_ts_is_recorded_for_purged_cv():
    s = make_series([1.0] * 200)
    c = BinaryContract(duration_s=10, entry_latency_ms=0)
    ls = label(s, np.array([1_000_000]), np.array([CALL]), c)
    assert ls.expiry_ts[0] == ls.entry_ts[0] + 10_000


def test_mismatched_input_lengths_rejected():
    s = make_series([1.0] * 10)
    with pytest.raises(ValueError):
        label(s, np.array([1_000_000, 1_001_000]), np.array([CALL]), BinaryContract())


def test_invalid_direction_rejected():
    s = make_series([1.0] * 10)
    with pytest.raises(ValueError):
        label(s, np.array([1_000_000]), np.array([0]), BinaryContract())


def test_empty_series_rejected():
    empty = TickSeries("X", "test", np.array([], dtype=np.int64), np.array([]))
    with pytest.raises(ValueError):
        label(empty, np.array([1]), np.array([CALL]), BinaryContract())
