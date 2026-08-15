import numpy as np
import pytest

from pobot.model.logistic import ConstantSignal, LogisticSignal


def separable(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
    return X, y


def test_learns_a_separable_boundary():
    X, y = separable()
    m = LogisticSignal(l2=0.01).fit(X, y)
    assert m.converged_
    assert ((m.predict_proba(X) > 0.5) == (y > 0.5)).mean() > 0.95


def test_coefficients_follow_the_true_signal():
    X, y = separable()
    m = LogisticSignal(l2=0.01).fit(X, y)
    assert m.coef_[0] > m.coef_[1] > 0
    assert abs(m.coef_[2]) < abs(m.coef_[1])  # the irrelevant feature stays small


def test_probabilities_are_valid():
    X, y = separable()
    p = LogisticSignal().fit(X, y).predict_proba(X)
    assert np.all((p >= 0) & (p <= 1))
    assert np.all(np.isfinite(p))


def test_pure_noise_yields_no_predictive_power():
    """On unpredictable data the model must not manufacture confidence."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 4))
    y = (rng.random(2000) > 0.5).astype(float)
    p = LogisticSignal(l2=1.0).fit(X, y).predict_proba(X)
    assert abs(p.mean() - 0.5) < 0.05
    assert p.std() < 0.10


def test_standardisation_uses_training_statistics_only():
    """Scaling on the full dataset leaks test distribution into training — a
    lookahead that hides in preprocessing and survives purged CV."""
    X, y = separable(1000)
    train, test = slice(0, 500), slice(500, None)
    m = LogisticSignal().fit(X[train], y[train])
    np.testing.assert_allclose(m.mean_, X[train].mean(axis=0))
    np.testing.assert_allclose(m.scale_, X[train].std(axis=0))
    # Shifting only the test rows must not alter the stored scaler.
    shifted = X.copy()
    shifted[test] += 100.0
    assert m.predict_proba(shifted[test]).mean() != m.predict_proba(X[test]).mean()
    np.testing.assert_allclose(m.mean_, X[train].mean(axis=0))


def test_constant_feature_does_not_blow_up():
    X, y = separable()
    X[:, 2] = 7.0  # zero variance
    m = LogisticSignal().fit(X, y)
    assert np.all(np.isfinite(m.predict_proba(X)))


def test_stronger_regularisation_shrinks_coefficients():
    X, y = separable()
    weak = LogisticSignal(l2=0.01).fit(X, y)
    strong = LogisticSignal(l2=1000.0).fit(X, y)
    assert np.abs(strong.coef_).sum() < np.abs(weak.coef_).sum()


def test_heavy_regularisation_degrades_to_the_base_rate():
    """Not to 0.5 — the intercept is unpenalised, so it keeps the class balance."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(2000, 3))
    y = (rng.random(2000) < 0.7).astype(float)
    p = LogisticSignal(l2=1e6).fit(X, y).predict_proba(X)
    assert p.mean() == pytest.approx(0.7, abs=0.03)


def test_predicting_before_fitting_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        LogisticSignal().predict_proba(np.zeros((3, 2)))


@pytest.mark.parametrize(
    "X,y,match",
    [
        (np.zeros((3, 2)), np.array([0.0, 1.0]), "same length"),
        (np.zeros((3, 2)), np.array([0.0, 1.0, 2.0]), "only 0 and 1"),
        (np.zeros((0, 2)), np.zeros(0), "empty"),
        (np.full((3, 2), np.nan), np.array([0.0, 1.0, 0.0]), "non-finite"),
    ],
)
def test_invalid_training_data_rejected(X, y, match):
    with pytest.raises(ValueError, match=match):
        LogisticSignal().fit(X, y)


def test_negative_l2_rejected():
    with pytest.raises(ValueError):
        LogisticSignal(l2=-1.0)


def test_constant_signal_predicts_the_base_rate():
    y = np.array([1.0] * 70 + [0.0] * 30)
    m = ConstantSignal().fit(np.zeros((100, 2)), y)
    assert m.predict_proba(np.zeros((5, 2))).tolist() == [0.7] * 5
