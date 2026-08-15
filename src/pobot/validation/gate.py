"""The go/no-go gate.

A backtest produces a win rate. This module answers the only question that
matters: is that win rate *reliably* above break-even, or is it noise?

The bar is high on purpose, for three reasons.

Small edges need large samples. At a 92% payout (break-even 52.08%), a one-sided
test at 95% confidence and 80% power needs roughly:

    true 53%  ->  18,400 trades
    true 54%  ->   4,200
    true 55%  ->   1,800
    true 56%  ->   1,000

The cost explodes as the edge shrinks, because required n scales with the
inverse square of the gap to break-even. A strategy showing 58% over 200 trades
has a 95% interval spanning roughly 51%-65% — consistent with a strong edge and
consistent with nothing. Both explanations survive, so the result decides
nothing. Use `required_sample_size` to check your data can answer the question
*before* running the experiment.

Testing many strategies breaks naive p-values. Try 100 variants at alpha=0.05
and about 5 will look significant on pure chance. Grid-searching indicator
parameters *is* testing hundreds of strategies, so `n_trials` applies a
Šidák correction. Set it honestly: it is the number of configurations you
evaluated, not the number you are reporting.

Overlapping trades inflate effective sample size. 10,000 heavily overlapping
60-second trades carry far less information than 10,000 independent ones. Pass
`overlap` from `purged_cv.overlap_fraction` and the effective n is discounted
accordingly.

The output is a pass/fail with reasons. Failing is the expected outcome, and the
correct response to a fail is to discard the strategy — not to re-tune it on the
same data until it passes, which is precisely the multiple-testing problem the
correction above exists to price in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from scipy import stats


def break_even_rate(payout: float) -> float:
    if payout <= 0:
        raise ValueError("payout must be positive")
    return 1.0 / (1.0 + payout)


def wilson_interval(wins: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation, which misbehaves for small n and
    for proportions near 0 or 1 — exactly the regimes an early-stage strategy
    evaluation lives in.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def required_sample_size(
    p0: float, p1: float, *, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Trades needed to distinguish a true rate `p1` from break-even `p0`.

    One-sided test of a single proportion. Use this *before* running an
    experiment to check the data you have can answer the question at all.
    """
    if not 0 < p0 < 1 or not 0 < p1 < 1:
        raise ValueError("p0 and p1 must be strictly between 0 and 1")
    if p1 <= p0:
        raise ValueError("p1 must exceed p0 — there is no edge to detect otherwise")
    z_a = stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(power)
    num = z_a * math.sqrt(p0 * (1 - p0)) + z_b * math.sqrt(p1 * (1 - p1))
    return int(math.ceil((num / (p1 - p0)) ** 2))


def effective_n(n: int, overlap: float) -> int:
    """Discount sample size for label overlap.

    `overlap` is the mean fraction of other samples each sample's life overlaps
    (see purged_cv.overlap_fraction). Deliberately conservative: overlapping
    trades share outcome information, so treating them as independent is the
    optimistic error.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    return max(1, int(n * (1.0 - overlap)))


@dataclass
class GateResult:
    passed: bool
    win_rate: float
    break_even: float
    n: int
    n_effective: int
    wins: int
    p_value: float
    alpha_adjusted: float
    ci_low: float
    ci_high: float
    ev_at_point: float
    ev_at_lower_bound: float
    required_n: Optional[int]
    reasons: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            f"{'PASS' if self.passed else 'FAIL'}  "
            f"win rate {self.win_rate:.2%} vs break-even {self.break_even:.2%}",
            f"  trades          {self.n} (effective {self.n_effective})",
            f"  95% CI          [{self.ci_low:.2%}, {self.ci_high:.2%}]",
            f"  p-value         {self.p_value:.4g} (alpha {self.alpha_adjusted:.4g})",
            f"  EV/trade        {self.ev_at_point:+.3%} at point estimate",
            f"  EV/trade        {self.ev_at_lower_bound:+.3%} at CI lower bound",
        ]
        if self.required_n:
            lines.append(f"  n for detection {self.required_n}")
        for r in self.reasons:
            lines.append(f"  - {r}")
        return "\n".join(lines)


def evaluate(
    wins: int,
    n: int,
    payout: float,
    *,
    alpha: float = 0.05,
    n_trials: int = 1,
    overlap: float = 0.0,
    min_trades: int = 1000,
    require_positive_lower_bound: bool = True,
) -> GateResult:
    """Decide whether an out-of-sample result clears the bar.

    Args:
        wins: Winning trades, ties excluded.
        n: Decided trades (wins + losses), ties excluded.
        payout: Contract payout fraction.
        alpha: Base significance level, before multiple-testing correction.
        n_trials: How many strategy variants you evaluated. Every parameter
            combination you tried counts, including the ones you discarded.
        overlap: Mean label overlap from purged_cv.overlap_fraction.
        min_trades: Floor on effective sample size. Below this, no win rate is
            informative regardless of how good it looks.
        require_positive_lower_bound: Require the *lower* confidence bound to
            beat break-even, not just the point estimate. This is the difference
            between "probably profitable" and "profitable unless I was unlucky in
            a way the data cannot rule out".

    Run this on out-of-sample fold results only. Applied in-sample it measures
    how well you fit noise.
    """
    if n < 0 or wins < 0 or wins > n:
        raise ValueError("require 0 <= wins <= n")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")

    be = break_even_rate(payout)
    reasons: list[str] = []

    if n == 0:
        return GateResult(
            passed=False, win_rate=float("nan"), break_even=be, n=0, n_effective=0,
            wins=0, p_value=1.0, alpha_adjusted=alpha, ci_low=float("nan"),
            ci_high=float("nan"), ev_at_point=float("nan"),
            ev_at_lower_bound=float("nan"), required_n=None,
            reasons=["no decided trades to evaluate"],
        )

    wr = wins / n
    n_eff = effective_n(n, overlap)
    # Scale wins with n so the tested proportion is preserved under discounting.
    wins_eff = int(round(wr * n_eff))

    # Šidák: less conservative than Bonferroni, same purpose.
    alpha_adj = 1 - (1 - alpha) ** (1 / n_trials)

    p_value = float(
        stats.binomtest(wins_eff, n_eff, be, alternative="greater").pvalue
    )
    ci_low, ci_high = wilson_interval(wins_eff, n_eff, conf=1 - alpha_adj)

    ev_point = wr * (1 + payout) - 1
    ev_low = ci_low * (1 + payout) - 1

    req_n = None
    if wr > be:
        try:
            req_n = required_sample_size(be, wr, alpha=alpha_adj)
        except ValueError:
            req_n = None

    if n_eff < min_trades:
        reasons.append(
            f"effective sample {n_eff} below minimum {min_trades} — "
            f"cannot distinguish edge from noise"
        )
    if wr <= be:
        reasons.append(
            f"win rate {wr:.2%} does not exceed break-even {be:.2%} — negative EV"
        )
    if p_value > alpha_adj:
        reasons.append(
            f"not significant: p={p_value:.4g} > alpha={alpha_adj:.4g}"
            + (f" (corrected for {n_trials} trials)" if n_trials > 1 else "")
        )
    if require_positive_lower_bound and ev_low <= 0:
        reasons.append(
            f"CI lower bound {ci_low:.2%} implies EV {ev_low:+.3%} — "
            f"edge not established with confidence"
        )
    if req_n and n_eff < req_n:
        reasons.append(f"underpowered: {req_n} trades needed to confirm this rate")

    return GateResult(
        passed=not reasons,
        win_rate=wr,
        break_even=be,
        n=n,
        n_effective=n_eff,
        wins=wins,
        p_value=p_value,
        alpha_adjusted=alpha_adj,
        ci_low=ci_low,
        ci_high=ci_high,
        ev_at_point=ev_point,
        ev_at_lower_bound=ev_low,
        required_n=req_n,
        reasons=reasons,
    )
