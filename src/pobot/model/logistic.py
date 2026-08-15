"""L2-regularised logistic regression.

Chosen over anything fancier on purpose. The signal being hunted is worth at
most a couple of percentage points of win rate; a high-capacity model on a
signal that weak mainly learns the noise faster, and then the cross-validation
has to spend its statistical power detecting the overfit instead of measuring
the edge. Start here, and only reach for something bigger once a linear model
has shown there is anything to find.

Standardisation statistics are computed in `fit` and reused in `predict_proba`.
That ordering is the whole point: computing them over the full dataset would
leak test-fold distribution into training, which is a quiet form of lookahead
that survives purged cross-validation because it hides in the preprocessing
rather than in the labels.

The interface is deliberately the scikit-learn subset (`fit`, `predict_proba`),
so a gradient booster can be dropped in without touching the study code.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class LogisticSignal:
    """Binary classifier predicting P(the 'up' outcome).

    Args:
        l2: Ridge penalty on the coefficients. The intercept is never penalised,
            so a heavily regularised model degrades to predicting the base rate
            rather than to predicting 0.5.
        max_iter: L-BFGS iteration cap.
    """

    def __init__(self, l2: float = 1.0, max_iter: int = 500) -> None:
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        self.l2 = l2
        self.max_iter = max_iter
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.converged_: bool = False

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.scale_ is not None
        return (X - self.mean_) / self.scale_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticSignal":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if len(X) == 0:
            raise ValueError("cannot fit on an empty dataset")
        if not np.all(np.isin(y, (0.0, 1.0))):
            raise ValueError("y must contain only 0 and 1")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values")

        # Train-fold statistics only. See module docstring.
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        Z = self._standardise(X)

        n, d = Z.shape

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            w, b = params[:d], params[d]
            z = Z @ w + b
            # logaddexp(0, z) - y*z is the numerically stable cross-entropy.
            loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
            loss += 0.5 * self.l2 * float(w @ w) / n

            resid = expit(z) - y
            gw = Z.T @ resid / n + self.l2 * w / n
            gb = float(resid.mean())
            return loss, np.concatenate([gw, [gb]])

        res = minimize(
            objective, np.zeros(d + 1), jac=True, method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        self.coef_ = res.x[:d]
        self.intercept_ = float(res.x[d])
        self.converged_ = bool(res.success)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        return self._standardise(np.asarray(X, dtype=np.float64)) @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(y = 1) for each row."""
        return expit(self.decision_function(X))


class ConstantSignal:
    """Always predicts the training base rate. The null model.

    A real model must beat this, and the comparison is more informative than it
    sounds: on a genuinely unpredictable series, a fitted model's out-of-sample
    accuracy converges to the base rate, so "we beat 50%" often just means the
    class balance was not 50/50.
    """

    def __init__(self) -> None:
        self.rate_: float = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConstantSignal":
        y = np.asarray(y, dtype=np.float64)
        self.rate_ = float(y.mean()) if len(y) else 0.5
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.rate_, dtype=np.float64)
