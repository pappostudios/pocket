import numpy as np
import pytest

from pobot.validation.purged_cv import PurgedWalkForward, overlap_fraction


def make_samples(n=500, step_ms=1000, life_ms=60_000, start=1_000_000):
    t0 = np.arange(start, start + n * step_ms, step_ms, dtype=np.int64)
    return t0, t0 + life_ms


def test_no_training_sample_survives_into_the_test_window():
    """The core guarantee: nothing in train is still live when test begins."""
    t0, t1 = make_samples()
    cv = PurgedWalkForward(n_splits=5, min_train=10)
    folds = list(cv.split(t0, t1))
    assert folds
    for fold in folds:
        assert (t1[fold.train_idx] < fold.test_start_ms).all()


def test_training_data_always_precedes_test_data():
    t0, t1 = make_samples()
    for fold in PurgedWalkForward(n_splits=5, min_train=10).split(t0, t1):
        assert fold.train_idx.max() < fold.test_idx.min()


def test_train_and_test_never_intersect():
    t0, t1 = make_samples()
    for fold in PurgedWalkForward(n_splits=4, min_train=10).split(t0, t1):
        assert not set(fold.train_idx) & set(fold.test_idx)


def test_overlapping_labels_are_actually_purged():
    """Long-lived labels overlap the test window and must be removed."""
    t0, t1 = make_samples(n=400, step_ms=1000, life_ms=120_000)
    folds = list(PurgedWalkForward(n_splits=4, min_train=5).split(t0, t1))
    assert sum(f.n_purged for f in folds) > 0


def test_short_labels_need_little_purging():
    t0, t1 = make_samples(n=400, step_ms=10_000, life_ms=1_000)
    folds = list(PurgedWalkForward(n_splits=4, min_train=5).split(t0, t1))
    assert sum(f.n_purged for f in folds) <= len(folds)


def test_rolling_window_bounds_training_size():
    t0, t1 = make_samples(n=600)
    rolling = list(PurgedWalkForward(n_splits=5, expanding=False, min_train=10).split(t0, t1))
    expanding = list(PurgedWalkForward(n_splits=5, expanding=True, min_train=10).split(t0, t1))
    assert max(len(f.train_idx) for f in rolling) <= max(
        len(f.train_idx) for f in expanding
    )


def test_overlap_fraction_zero_for_disjoint_labels():
    t0 = np.arange(0, 100_000, 10_000, dtype=np.int64)
    assert overlap_fraction(t0, t0 + 1_000) == pytest.approx(0.0)


def test_overlap_fraction_positive_for_overlapping_labels():
    t0 = np.arange(0, 100_000, 1_000, dtype=np.int64)
    assert overlap_fraction(t0, t0 + 60_000) > 0.0


def test_heavier_overlap_scores_higher():
    t0 = np.arange(0, 200_000, 1_000, dtype=np.int64)
    assert overlap_fraction(t0, t0 + 60_000) > overlap_fraction(t0, t0 + 5_000)


@pytest.mark.parametrize("bad", [{"n_splits": 1}, {"embargo_pct": 1.0}, {"embargo_pct": -0.1}])
def test_invalid_configuration_rejected(bad):
    with pytest.raises(ValueError):
        PurgedWalkForward(**{"n_splits": 5, **bad})


def test_unsorted_entry_times_rejected():
    t0 = np.array([5, 1, 3], dtype=np.int64)
    with pytest.raises(ValueError):
        list(PurgedWalkForward(n_splits=2, min_train=1).split(t0, t0 + 10))


def test_expiry_before_entry_rejected():
    t0 = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError):
        list(PurgedWalkForward(n_splits=2, min_train=1).split(t0, t0 - 10))


def test_too_few_samples_rejected():
    t0 = np.array([1, 2], dtype=np.int64)
    with pytest.raises(ValueError):
        list(PurgedWalkForward(n_splits=5).split(t0, t0 + 1))
