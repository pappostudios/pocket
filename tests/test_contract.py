import pytest

from pobot.backtest.contract import CALL, PUT, PAYOUT_TABLE, BinaryContract


def test_break_even_rate_matches_published_table():
    for payout, expected in PAYOUT_TABLE.items():
        assert BinaryContract(payout=payout).break_even_rate == pytest.approx(
            expected, abs=1e-4
        )


def test_coin_flip_at_92_percent_loses_4_percent_per_trade():
    """The number that motivates the whole project."""
    c = BinaryContract(payout=0.92)
    assert c.expected_value(0.50) == pytest.approx(-0.04)


def test_expected_value_is_zero_at_break_even():
    for payout in (0.70, 0.80, 0.85, 0.92):
        c = BinaryContract(payout=payout)
        assert c.expected_value(c.break_even_rate) == pytest.approx(0.0, abs=1e-12)


def test_settlement_directions():
    c = BinaryContract(payout=0.92)
    assert c.settle(1.10, 1.11, CALL) == 0.92
    assert c.settle(1.10, 1.09, CALL) == -1.0
    assert c.settle(1.10, 1.09, PUT) == 0.92
    assert c.settle(1.10, 1.11, PUT) == -1.0


def test_tie_rules():
    assert BinaryContract(tie_rule="refund").settle(1.1, 1.1, CALL) == 0.0
    assert BinaryContract(tie_rule="loss").settle(1.1, 1.1, CALL) == -1.0


def test_relative_expiry():
    c = BinaryContract(duration_s=60, expiry_mode="relative")
    assert c.expiry_ts(1_000_000) == 1_000_000 + 60_000


def test_clock_expiry_snaps_to_next_boundary():
    c = BinaryContract(duration_s=60, expiry_mode="clock")
    assert c.expiry_ts(1_000_017_000) == 1_000_020_000


def test_clock_expiry_on_a_boundary_takes_the_next_one():
    """A zero-life contract would settle at its own strike — always a tie."""
    c = BinaryContract(duration_s=60, expiry_mode="clock")
    assert c.expiry_ts(1_000_020_000) == 1_000_080_000


def test_entry_latency_shifts_the_strike_time():
    c = BinaryContract(entry_latency_ms=250)
    assert c.entry_ts(1_000_000) == 1_000_250


def test_spread_is_always_adverse():
    c = BinaryContract(spread=0.0002)
    assert c.effective_strike(1.1000, CALL) == pytest.approx(1.1002)
    assert c.effective_strike(1.1000, PUT) == pytest.approx(1.0998)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"payout": 0},
        {"payout": -0.5},
        {"duration_s": 0},
        {"entry_latency_ms": -1},
        {"spread": -0.001},
    ],
)
def test_invalid_contracts_rejected(kwargs):
    with pytest.raises(ValueError):
        BinaryContract(**kwargs)
