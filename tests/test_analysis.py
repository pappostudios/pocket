import numpy as np
import pytest

from pobot.analysis.fingerprint import (
    fingerprint,
    grid_quantisation,
    ljung_box,
    runs_test,
    variance_ratio,
)
from pobot.analysis.lag import estimate_lag, resample_last
from pobot.data.store import TickSeries, ticks_to_series
from pobot.feeds.synthetic import LaggedFeed, MeanRevertingFeed, RandomWalkFeed


def series_from(feed, n, symbol="X"):
    return ticks_to_series(feed.generate(n), symbol, feed.name)


# --- Lag estimation ---------------------------------------------------------

def test_resample_uses_last_price_at_or_before():
    s = TickSeries("X", "t", np.array([0, 1000, 2000], dtype=np.int64),
                   np.array([1.0, 2.0, 3.0]))
    out = resample_last(s, np.array([0, 500, 1000, 2500], dtype=np.int64))
    assert list(out) == [1.0, 1.0, 2.0, 3.0]


def test_resample_is_nan_before_the_series_starts():
    s = TickSeries("X", "t", np.array([1000], dtype=np.int64), np.array([1.0]))
    assert np.isnan(resample_last(s, np.array([0], dtype=np.int64))[0])


@pytest.mark.parametrize("lag_ms", [200, 300, 500])
def test_known_lag_is_recovered(lag_ms):
    """The core capability: a delayed copy must be identified as delayed."""
    ref = series_from(RandomWalkFeed(["X"], interval_ms=100, seed=9), 8000)
    broker = series_from(
        LaggedFeed(RandomWalkFeed(["X"], interval_ms=100, seed=9), lag_ms=lag_ms), 8000
    )
    est = estimate_lag(broker, ref, grid_ms=100, max_lag_ms=1500)
    assert est.peak_lag_ms == lag_ms
    assert est.peak_corr > 0.9
    assert est.significant
    assert est.broker_lags


def test_independent_feeds_show_no_tradeable_lag():
    """Scanning 31 offsets on unrelated noise must not manufacture an edge."""
    ref = series_from(RandomWalkFeed(["X"], interval_ms=100, seed=9), 8000)
    other = series_from(RandomWalkFeed(["X"], interval_ms=100, seed=77), 8000)
    est = estimate_lag(other, ref, grid_ms=100, max_lag_ms=1500)
    assert not est.broker_lags
    assert abs(est.peak_corr) < 0.1


def test_identical_feeds_peak_at_zero_lag():
    ref = series_from(RandomWalkFeed(["X"], interval_ms=100, seed=3), 5000)
    est = estimate_lag(ref, ref, grid_ms=100, max_lag_ms=1000)
    assert est.peak_lag_ms == 0
    assert not est.broker_lags  # moving together is not a delay


def test_too_little_overlap_is_rejected():
    short = series_from(RandomWalkFeed(["X"], interval_ms=100, seed=1), 50)
    with pytest.raises(ValueError, match="too short"):
        estimate_lag(short, short, grid_ms=100, max_lag_ms=3000)


@pytest.mark.parametrize("kwargs", [{"grid_ms": 0}, {"max_lag_ms": 50}])
def test_invalid_lag_parameters_rejected(kwargs):
    s = series_from(RandomWalkFeed(["X"], interval_ms=100, seed=1), 5000)
    with pytest.raises(ValueError):
        estimate_lag(s, s, **{"grid_ms": 100, "max_lag_ms": 1000, **kwargs})


# --- Variance ratio ---------------------------------------------------------

@pytest.mark.parametrize("q", [2, 5, 10])
def test_variance_ratio_variance_matches_homoskedastic_form(q):
    """Calibration check on iid data.

    Under homoskedasticity the Lo-MacKinlay robust variance must reduce to
    2(2q-1)(q-1)/(3qn). A scaling error here silently destroys the test's power
    while leaving the VR statistic itself looking correct.
    """
    r = np.random.default_rng(0).normal(0, 1, 200_000)
    res = variance_ratio(r, q)
    z = float(res.detail.split("z=")[1])
    theta_empirical = ((res.statistic - 1) / z) ** 2
    closed_form = 2 * (2 * q - 1) * (q - 1) / (3 * q * len(r))
    assert theta_empirical == pytest.approx(closed_form, rel=0.05)


def test_variance_ratio_is_one_for_a_random_walk():
    r = np.random.default_rng(1).normal(0, 1, 20_000)
    assert variance_ratio(r, 5).statistic == pytest.approx(1.0, abs=0.05)


def test_variance_ratio_detects_mean_reversion():
    s = series_from(MeanRevertingFeed(["X"], kappa=0.2, seed=5), 6000)
    res = variance_ratio(np.diff(np.log(s.price)), 10)
    assert res.statistic < 0.6
    assert res.p_value < 1e-6


def test_variance_ratio_rejects_impossible_horizons():
    with pytest.raises(ValueError):
        variance_ratio(np.random.default_rng(0).normal(size=1000), 1)
    with pytest.raises(ValueError):
        variance_ratio(np.random.default_rng(0).normal(size=10), 5)


# --- Other fingerprint tests ------------------------------------------------

def test_ljung_box_detects_autocorrelation():
    rng = np.random.default_rng(2)
    x = rng.normal(size=5000)
    ar = np.zeros(5000)
    for i in range(1, 5000):
        ar[i] = -0.4 * ar[i - 1] + x[i]
    assert ljung_box(ar).p_value < 1e-6
    assert ljung_box(rng.normal(size=5000)).p_value > 0.01


def test_grid_quantisation_flags_a_fixed_tick():
    on_grid = 1.0 + np.cumsum(np.random.default_rng(3).integers(-3, 4, 2000)) * 1e-4
    assert grid_quantisation(on_grid).statistic > 0.99
    continuous = 1.0 + np.cumsum(np.random.default_rng(3).normal(0, 1e-4, 2000))
    assert grid_quantisation(continuous).statistic < 0.5


def test_runs_test_detects_alternation():
    alternating = np.array([1.0, -1.0] * 1000)
    assert runs_test(alternating).p_value < 1e-6
    assert runs_test(np.random.default_rng(4).normal(size=2000)).p_value > 0.01


# --- Full battery -----------------------------------------------------------

def test_random_walk_is_not_flagged_as_synthetic():
    """The null case. False positives here would send you chasing noise."""
    fp = fingerprint(series_from(RandomWalkFeed(["X"], seed=5), 6000))
    assert not fp.looks_synthetic
    assert fp.anomalies == []


def test_mean_reverting_series_is_flagged():
    fp = fingerprint(series_from(MeanRevertingFeed(["X"], kappa=0.2, seed=5), 6000))
    assert fp.looks_synthetic
    assert len(fp.anomalies) >= 2


def test_fingerprint_corrects_for_the_number_of_tests():
    fp = fingerprint(series_from(RandomWalkFeed(["X"], seed=1), 6000), alpha=0.05)
    assert fp.alpha_adjusted < 0.05
    assert len(fp.tests) == 6


def test_fingerprint_needs_enough_data():
    with pytest.raises(ValueError, match="at least 200"):
        fingerprint(series_from(RandomWalkFeed(["X"], seed=1), 100))


def test_fingerprint_report_renders():
    assert "random walk" in fingerprint(
        series_from(RandomWalkFeed(["X"], seed=1), 1000)
    ).report()
