import numpy as np
import pytest

from pobot.backtest.engine import MarketView
from pobot.data.store import TickSeries, ticks_to_series
from pobot.features.core import build_features, compute_row, feature_names
from pobot.feeds.synthetic import RandomWalkFeed


def make_series(n=500, seed=1):
    feed = RandomWalkFeed(["X"], interval_ms=1000, seed=seed)
    return ticks_to_series(feed.generate(n), "X", feed.name)


def test_vectorised_features_match_the_causal_reference():
    """The load-bearing test for this module.

    `compute_row` reads through `MarketView`, which physically cannot return
    future data. If the vectorised builder agrees with it exactly at every
    sampled index, the vectorised path is not reaching forward either. An
    off-by-one in the permissive direction is invisible in the numbers and
    fatal to the study, so it gets a proof rather than a comment.
    """
    series = make_series(600)
    windows = (10, 30, 60)
    fs = build_features(series, windows)

    for row in range(0, len(fs), 37):
        i = int(fs.index[row])
        expected = compute_row(MarketView(series, i), windows)
        np.testing.assert_allclose(fs.matrix[row], expected, rtol=1e-12, atol=1e-12)


def test_features_start_only_where_history_is_complete():
    series = make_series(300)
    fs = build_features(series, (10, 30, 60))
    assert int(fs.index[0]) == 59  # max window - 1
    assert int(fs.index[-1]) == len(series) - 1
    assert len(fs) == len(series) - 59


def test_matrix_shape_matches_declared_names():
    fs = build_features(make_series(300), (10, 30))
    assert fs.matrix.shape == (len(fs), len(fs.names))
    assert fs.names == feature_names((10, 30))
    assert len(set(fs.names)) == len(fs.names)


def test_all_features_are_finite():
    fs = build_features(make_series(500), (10, 30, 60))
    assert np.all(np.isfinite(fs.matrix))


def test_flat_series_degrades_gracefully():
    """Zero variance must yield defined values, not NaN poisoning the model."""
    flat = TickSeries("X", "t", np.arange(0, 300_000, 1000, dtype=np.int64),
                      np.full(300, 1.1))
    fs = build_features(flat, (10, 30))
    assert np.all(np.isfinite(fs.matrix))
    z = fs.matrix[:, fs.names.index("zscore_10")]
    assert np.all(z == 0.0)
    rp = fs.matrix[:, fs.names.index("rangepos_10")]
    assert np.all(rp == 0.5)  # mid-range when there is no range


def test_rangepos_is_one_at_a_window_high():
    rising = TickSeries("X", "t", np.arange(0, 100_000, 1000, dtype=np.int64),
                        np.linspace(1.0, 1.1, 100))
    fs = build_features(rising, (10,))
    assert np.allclose(fs.matrix[:, fs.names.index("rangepos_10")], 1.0)


def test_upfrac_is_one_in_a_monotonic_rise():
    rising = TickSeries("X", "t", np.arange(0, 100_000, 1000, dtype=np.int64),
                        np.linspace(1.0, 1.1, 100))
    fs = build_features(rising, (10,))
    assert np.allclose(fs.matrix[:, fs.names.index("upfrac_10")], 1.0)


def test_subset_keeps_rows_and_index_aligned():
    fs = build_features(make_series(400), (10, 30))
    rows = np.arange(0, len(fs), 5)
    sub = fs.subset(rows)
    assert len(sub) == len(rows)
    np.testing.assert_array_equal(sub.index, fs.index[rows])
    np.testing.assert_array_equal(sub.matrix, fs.matrix[rows])


@pytest.mark.parametrize("windows", [(), (1,), (0,)])
def test_invalid_windows_rejected(windows):
    with pytest.raises(ValueError):
        build_features(make_series(300), windows)


def test_series_shorter_than_the_window_is_rejected():
    with pytest.raises(ValueError, match="need more than"):
        build_features(make_series(30), (60,))
