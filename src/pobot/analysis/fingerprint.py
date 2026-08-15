"""Statistical fingerprinting of a price series.

Tests the second hypothesis worth investigating: that broker-generated "OTC"
assets — the ones quoted at weekends, when the real market is shut — are
synthetic series with structure a real market would have arbitraged away.

A real, liquid FX series is close to a martingale on short horizons. Returns are
near-uncorrelated, variance grows roughly linearly with the sampling interval,
and prices are not confined to a coarse grid. A generated series often fails at
least one of those, because whoever wrote the generator was aiming for
plausible-looking, not statistically indistinguishable.

Four independent tests, chosen because they fail in different ways:

* **Variance ratio** (Lo-MacKinlay). VR(q) = Var(q-period returns) / (q x
  Var(1-period returns)). Exactly 1 for a random walk; below 1 under mean
  reversion, above 1 under trending. The z-statistic here is the
  heteroskedasticity-robust form, so ordinary volatility clustering does not
  by itself trip it.

* **Return autocorrelation** with a Ljung-Box portmanteau test. Detects
  short-horizon predictability that variance ratio can miss when effects at
  different lags cancel.

* **Price grid quantisation.** Generators frequently emit prices on a fixed
  tick. A series where nearly every increment is a multiple of one constant is
  not a market observation.

* **Runs test** on the sign of returns. Detects streakiness or alternation that
  leaves variance and autocorrelation untouched.

What a positive result means, precisely: the series is *not* a random walk, so
some structure exists to model. It does not mean the structure is tradeable — a
detectable pattern still has to clear the payout, the spread, and entry latency.
Take a fingerprint hit as permission to run a proper study, not as a signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..data.store import TickSeries


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    detail: str = ""

    def significant(self, alpha: float) -> bool:
        return self.p_value < alpha


@dataclass
class Fingerprint:
    n_ticks: int
    n_returns: int
    alpha_adjusted: float
    tests: list[TestResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def anomalies(self) -> list[TestResult]:
        return [t for t in self.tests if t.significant(self.alpha_adjusted)]

    @property
    def looks_synthetic(self) -> bool:
        return bool(self.anomalies)

    def report(self) -> str:
        lines = [
            f"{self.n_ticks} ticks, {self.n_returns} returns "
            f"(alpha {self.alpha_adjusted:.4g}, corrected for {len(self.tests)} tests)",
            "",
        ]
        for t in self.tests:
            mark = "ANOMALY" if t.significant(self.alpha_adjusted) else "  ok   "
            lines.append(f"  [{mark}] {t.name:<28} stat={t.statistic:>8.3f} "
                         f"p={t.p_value:.3g}")
            if t.detail:
                lines.append(f"            {t.detail}")
        lines.append("")
        if self.looks_synthetic:
            lines.append(
                f"=> {len(self.anomalies)} test(s) reject the random-walk null. "
                "Structure exists to model."
            )
            lines.append(
                "   Structure is not an edge: it must still clear payout, spread, "
                "and entry latency. Run a full study before concluding anything."
            )
        else:
            lines.append(
                "=> Indistinguishable from a random walk on these tests. No "
                "exploitable structure found."
            )
        for n in self.notes:
            lines.append(f"   - {n}")
        return "\n".join(lines)


def variance_ratio(returns: np.ndarray, q: int) -> TestResult:
    """Lo-MacKinlay variance ratio test, heteroskedasticity-robust."""
    n = len(returns)
    if q < 2:
        raise ValueError("q must be at least 2")
    if n < q * 4:
        raise ValueError(f"need at least {q * 4} returns for q={q}, got {n}")

    mu = returns.mean()
    var1 = np.sum((returns - mu) ** 2) / (n - 1)
    if var1 == 0:
        return TestResult(f"variance ratio q={q}", 1.0, 1.0, "series is constant")

    # Overlapping q-period returns, with the Lo-MacKinlay small-sample correction.
    cs = np.cumsum(returns)
    q_rets = cs[q - 1 :] - np.concatenate(([0.0], cs[: len(cs) - q]))
    m = q * (n - q + 1) * (1 - q / n)
    varq = np.sum((q_rets - q * mu) ** 2) / m
    vr = varq / var1

    # Heteroskedasticity-robust variance of VR (Lo-MacKinlay 1988):
    #
    #   delta_j = sum_t (x_t - mu)^2 (x_{t-j} - mu)^2 / [ sum_t (x_t - mu)^2 ]^2
    #   theta   = sum_{j=1}^{q-1} [ 2(q-j)/q ]^2 * delta_j
    #
    # delta_j is already O(1/n) — the denominator is the *square* of the sum of
    # squared deviations, so no further scaling by n belongs here. Under
    # homoskedasticity this reduces to the textbook 2(2q-1)(q-1)/(3qn), which
    # test_variance_ratio_variance_matches_homoskedastic_form checks directly.
    dev2 = (returns - mu) ** 2
    denom = dev2.sum() ** 2
    theta = 0.0
    for j in range(1, q):
        num = np.sum(dev2[j:] * dev2[: n - j])
        delta = num / denom if denom > 0 else 0.0
        theta += ((2 * (q - j) / q) ** 2) * delta
    if theta <= 0:
        return TestResult(f"variance ratio q={q}", vr, 1.0, "degenerate variance")

    z = (vr - 1) / math.sqrt(theta)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    direction = "mean-reverting" if vr < 1 else "trending"
    return TestResult(
        f"variance ratio q={q}", vr, p,
        f"VR={vr:.3f} ({direction} vs random walk), z={z:.2f}",
    )


def ljung_box(returns: np.ndarray, lags: int = 10) -> TestResult:
    """Portmanteau test for autocorrelation up to `lags`."""
    n = len(returns)
    if n < lags * 4:
        raise ValueError(f"need at least {lags * 4} returns, got {n}")
    x = returns - returns.mean()
    denom = np.sum(x**2)
    if denom == 0:
        return TestResult(f"ljung-box lags={lags}", 0.0, 1.0, "series is constant")

    acf = [float(np.sum(x[k:] * x[: n - k]) / denom) for k in range(1, lags + 1)]
    q = n * (n + 2) * sum((r**2) / (n - k) for k, r in enumerate(acf, start=1))
    p = float(1 - stats.chi2.cdf(q, df=lags))
    strongest = max(range(len(acf)), key=lambda i: abs(acf[i]))
    return TestResult(
        f"ljung-box lags={lags}", q, p,
        f"strongest autocorrelation r={acf[strongest]:+.4f} at lag {strongest + 1}",
    )


def grid_quantisation(prices: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Detect prices confined to a coarse fixed tick.

    Reported as a proportion rather than a distribution test: a series where
    almost every increment is a multiple of one constant is not a market
    observation, and the evidence is the proportion itself.
    """
    diffs = np.abs(np.diff(prices))
    nz = diffs[diffs > 0]
    if len(nz) < 50:
        return TestResult("grid quantisation", 0.0, 1.0, "too few price changes")

    tick = float(np.min(nz))
    if tick <= 0:
        return TestResult("grid quantisation", 0.0, 1.0, "degenerate tick size")

    ratios = nz / tick
    on_grid = np.abs(ratios - np.round(ratios)) < 1e-6
    frac = float(on_grid.mean())

    # Float64 prices on a genuine decimal grid land near 1.0; a continuous
    # series does not. p is a decision marker, not a distributional quantity.
    p = 0.0 if frac > 0.99 else 1.0
    return TestResult(
        "grid quantisation", frac, p,
        f"{frac:.1%} of {len(nz)} moves are exact multiples of {tick:.3g}",
    )


def runs_test(returns: np.ndarray) -> TestResult:
    """Wald-Wolfowitz runs test on the sign of returns."""
    signs = np.sign(returns)
    nz = signs[signs != 0]
    n = len(nz)
    if n < 50:
        return TestResult("runs (sign)", 0.0, 1.0, "too few nonzero returns")

    n_pos = int((nz > 0).sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return TestResult("runs (sign)", 0.0, 0.0, "returns are single-signed")

    runs = 1 + int((nz[1:] != nz[:-1]).sum())
    exp = 2 * n_pos * n_neg / n + 1
    var = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1))
    if var <= 0:
        return TestResult("runs (sign)", float(runs), 1.0, "degenerate variance")

    z = (runs - exp) / math.sqrt(var)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    tendency = "streaky" if runs < exp else "alternating"
    return TestResult(
        "runs (sign)", float(runs), p,
        f"{runs} runs vs {exp:.1f} expected ({tendency}), z={z:.2f}",
    )


def fingerprint(
    series: TickSeries, *, alpha: float = 0.05, vr_horizons: tuple[int, ...] = (2, 5, 10)
) -> Fingerprint:
    """Run the full battery against a price series.

    The significance level is Šidák-corrected across the tests actually run,
    since asking several questions raises the chance one answers 'yes' by luck.
    """
    prices = np.asarray(series.price, dtype=np.float64)
    if len(prices) < 200:
        raise ValueError(f"need at least 200 ticks to fingerprint, got {len(prices)}")
    if np.any(prices <= 0):
        raise ValueError("prices must be positive to take log returns")

    returns = np.diff(np.log(prices))
    notes: list[str] = []

    tests: list[TestResult] = []
    for q in vr_horizons:
        try:
            tests.append(variance_ratio(returns, q))
        except ValueError as exc:
            notes.append(f"variance ratio q={q} skipped: {exc}")
    try:
        tests.append(ljung_box(returns))
    except ValueError as exc:
        notes.append(f"ljung-box skipped: {exc}")
    tests.append(grid_quantisation(prices))
    tests.append(runs_test(returns))

    alpha_adj = 1 - (1 - alpha) ** (1 / max(1, len(tests)))
    return Fingerprint(
        n_ticks=len(prices),
        n_returns=len(returns),
        alpha_adjusted=alpha_adj,
        tests=tests,
        notes=notes,
    )
