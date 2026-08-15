import pytest

from pobot.validation.gate import (
    break_even_rate,
    effective_n,
    evaluate,
    required_sample_size,
    wilson_interval,
)


def test_break_even_rates():
    assert break_even_rate(0.92) == pytest.approx(0.5208, abs=1e-4)
    assert break_even_rate(0.80) == pytest.approx(0.5556, abs=1e-4)


def test_detecting_54_percent_needs_about_four_thousand_trades():
    """One-sided, one-sample test against a known break-even rate."""
    n = required_sample_size(break_even_rate(0.92), 0.54)
    assert 3_800 < n < 4_600


def test_a_one_point_edge_costs_far_more_data_than_a_two_point_one():
    """Required n scales with the inverse square of the gap to break-even."""
    be = break_even_rate(0.92)
    assert required_sample_size(be, 0.53) > 4 * required_sample_size(be, 0.54)


def test_smaller_edges_need_more_data():
    be = break_even_rate(0.92)
    assert required_sample_size(be, 0.53) > required_sample_size(be, 0.56)


def test_no_sample_size_detects_a_nonexistent_edge():
    with pytest.raises(ValueError):
        required_sample_size(break_even_rate(0.92), 0.50)


def test_wilson_interval_contains_the_point_estimate():
    lo, hi = wilson_interval(550, 1000)
    assert lo < 0.55 < hi


def test_wilson_interval_narrows_with_more_data():
    small = wilson_interval(58, 100)
    large = wilson_interval(5800, 10000)
    assert (large[1] - large[0]) < (small[1] - small[0])


# --- Gate decisions ---------------------------------------------------------

def test_small_sample_fails_however_good_it_looks():
    """58% over 200 trades decides nothing — the interval spans both stories."""
    r = evaluate(wins=116, n=200, payout=0.92)
    assert not r.passed
    assert any("sample" in x for x in r.reasons)


def test_coin_flip_fails():
    r = evaluate(wins=5000, n=10_000, payout=0.92)
    assert not r.passed
    assert any("break-even" in x for x in r.reasons)


def test_win_rate_below_break_even_fails():
    r = evaluate(wins=5150, n=10_000, payout=0.92)  # 51.5% < 52.08%
    assert not r.passed
    assert r.ev_at_point < 0


def test_genuine_edge_on_a_large_sample_passes():
    r = evaluate(wins=5600, n=10_000, payout=0.92)  # 56%
    assert r.passed
    assert r.reasons == []
    assert r.ev_at_lower_bound > 0


def test_marginal_edge_fails_on_confidence_not_on_point_estimate():
    """52.5% beats break-even on average but not with confidence."""
    r = evaluate(wins=5250, n=10_000, payout=0.92)
    assert r.win_rate > r.break_even
    assert not r.passed


def test_multiple_testing_correction_raises_the_bar():
    """Grid searching 200 variants is testing 200 hypotheses."""
    once = evaluate(wins=5400, n=10_000, payout=0.92, n_trials=1)
    many = evaluate(wins=5400, n=10_000, payout=0.92, n_trials=200)
    assert many.alpha_adjusted < once.alpha_adjusted
    assert many.ci_low < once.ci_low


def test_overlap_discounts_effective_sample_size():
    clean = evaluate(wins=5400, n=10_000, payout=0.92, overlap=0.0)
    overlapped = evaluate(wins=5400, n=10_000, payout=0.92, overlap=0.9)
    assert overlapped.n_effective < clean.n_effective
    assert overlapped.p_value > clean.p_value


def test_lower_payout_demands_a_higher_win_rate():
    """54% is profitable at a 92% payout and loss-making at 80%."""
    good = evaluate(wins=10_800, n=20_000, payout=0.92)
    bad = evaluate(wins=10_800, n=20_000, payout=0.80)
    assert good.passed
    assert not bad.passed


def test_effective_n_discounting():
    assert effective_n(1000, 0.0) == 1000
    assert effective_n(1000, 0.5) == 500


def test_zero_trades_fails_cleanly():
    r = evaluate(wins=0, n=0, payout=0.92)
    assert not r.passed
    assert r.n == 0


def test_report_is_renderable():
    assert "win rate" in evaluate(wins=5600, n=10_000, payout=0.92).report()


@pytest.mark.parametrize("bad", [{"wins": -1, "n": 10}, {"wins": 11, "n": 10}])
def test_invalid_counts_rejected(bad):
    with pytest.raises(ValueError):
        evaluate(payout=0.92, **bad)
