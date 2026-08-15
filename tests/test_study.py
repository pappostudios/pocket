"""End-to-end study tests.

`test_random_walk_study_fails` is the most important test in the repository.
The study is the component that will actually decide whether real money gets
risked, and a driftless random walk is the one case where the correct answer is
known with certainty: there is no edge. If the study ever passes on it, every
other number the pipeline produces is worthless.
"""

import numpy as np
import pytest

from pobot.backtest.contract import BinaryContract
from pobot.data.store import ticks_to_series
from pobot.feeds.synthetic import MeanRevertingFeed, RandomWalkFeed
from pobot.model.logistic import ConstantSignal
from pobot.study import DEFAULT_MARGINS, _pick_margin, run_study

CONTRACT = BinaryContract(payout=0.92, duration_s=60, entry_latency_ms=250)


def series_from(feed, n=40_000):
    return ticks_to_series(feed.generate(n), "X", feed.name)


@pytest.fixture(scope="module")
def rw_study():
    s = series_from(RandomWalkFeed(["X"], interval_ms=1000, seed=4))
    return run_study(s, CONTRACT, n_splits=5, stride=5, min_gate_trades=200)


@pytest.fixture(scope="module")
def ou_study():
    s = series_from(MeanRevertingFeed(["X"], kappa=0.30, interval_ms=1000, seed=4))
    return run_study(s, CONTRACT, n_splits=5, stride=5, min_gate_trades=200)


# --- The two known answers --------------------------------------------------

def test_random_walk_study_fails(rw_study):
    """No edge exists on a driftless random walk, so none may be reported."""
    assert not rw_study.passed
    assert not rw_study.gate.passed
    assert rw_study.win_rate < CONTRACT.break_even_rate


def test_mean_reverting_study_passes(ou_study):
    """A study that can only ever say 'no' is not a test."""
    assert ou_study.passed
    assert ou_study.gate.passed
    assert ou_study.win_rate > CONTRACT.break_even_rate


# --- Guardrails -------------------------------------------------------------

def test_majority_baseline_is_reported_and_enforced(rw_study):
    """Beating 50% is not an edge — in a drifting market 'always call' does."""
    assert 0.5 <= rw_study.majority_win_rate <= 1.0
    assert not rw_study.beats_majority


def test_passing_requires_beating_the_majority_baseline(ou_study):
    assert ou_study.beats_majority
    assert ou_study.majority_win_rate < ou_study.win_rate


def test_a_null_model_never_passes():
    """ConstantSignal carries no information, so it must not clear the gate even
    on a series where a real edge exists."""
    s = series_from(MeanRevertingFeed(["X"], kappa=0.30, interval_ms=1000, seed=4))
    res = run_study(s, CONTRACT, n_splits=5, stride=5, min_gate_trades=200,
                    model_factory=ConstantSignal)
    assert not res.passed


def test_folds_are_walk_forward_and_purged(ou_study):
    assert len(ou_study.folds) >= 3
    assert all(f.n_train > 0 and f.n_test > 0 for f in ou_study.folds)
    assert sum(f.n_purged for f in ou_study.folds) > 0


def test_training_sets_grow_across_folds(ou_study):
    sizes = [f.n_train for f in ou_study.folds]
    assert sizes == sorted(sizes)


def test_overlap_is_measured_and_fed_to_the_gate(ou_study):
    assert 0.0 <= ou_study.label_overlap < 1.0
    assert ou_study.gate.n_effective <= ou_study.gate.n


def test_multiple_testing_correction_is_threaded_through():
    s = series_from(MeanRevertingFeed(["X"], kappa=0.30, interval_ms=1000, seed=4))
    once = run_study(s, CONTRACT, n_splits=5, stride=5, min_gate_trades=200, n_trials=1)
    many = run_study(s, CONTRACT, n_splits=5, stride=5, min_gate_trades=200, n_trials=500)
    assert many.gate.alpha_adjusted < once.gate.alpha_adjusted
    assert many.gate.ci_low < once.gate.ci_low


def test_report_renders_for_both_outcomes(rw_study, ou_study):
    assert "FAIL" in rw_study.report()
    assert "PASS" in ou_study.report()
    assert "majority baseline" in rw_study.report()


# --- Margin selection -------------------------------------------------------

def test_margin_selection_prefers_a_real_edge():
    """A confident, accurate model should be traded at a permissive margin."""
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.5).astype(float)
    proba = np.where(y == 1, 0.8, 0.2)  # perfectly informative
    assert _pick_margin(proba, y, 0.92, DEFAULT_MARGINS, 20) <= 0.08


def test_margin_selection_ignores_tiny_trade_counts():
    """A margin taking three trades can post a spectacular rate on nothing."""
    proba = np.full(1000, 0.5)
    proba[:3] = 0.99  # three extremely confident, correct calls
    y = np.zeros(1000)
    y[:3] = 1.0
    assert _pick_margin(proba, y, 0.92, DEFAULT_MARGINS, min_trades=100) == DEFAULT_MARGINS[0]


def test_margin_is_chosen_per_fold_from_training_data(ou_study):
    assert all(f.margin in DEFAULT_MARGINS for f in ou_study.folds)


# --- Input validation -------------------------------------------------------

def test_short_series_is_rejected():
    s = series_from(RandomWalkFeed(["X"], interval_ms=1000, seed=1), 500)
    with pytest.raises(ValueError):
        run_study(s, CONTRACT, n_splits=5)


def test_invalid_stride_rejected():
    s = series_from(RandomWalkFeed(["X"], interval_ms=1000, seed=1), 5000)
    with pytest.raises(ValueError, match="stride"):
        run_study(s, CONTRACT, stride=0)


def test_features_stay_aligned_with_labels_after_drops():
    """Labelling drops unsettleable contracts; features must be re-indexed to
    match, or every row is paired with another row's outcome."""
    s = series_from(RandomWalkFeed(["X"], interval_ms=1000, seed=8), 20_000)
    res = run_study(s, CONTRACT, n_splits=4, stride=5, min_train=100, min_gate_trades=100)
    # The final contract cannot settle within the data, so at least one row drops.
    assert res.n_samples < len(s)
    assert res.trades <= res.n_samples
