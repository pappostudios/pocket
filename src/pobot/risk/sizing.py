"""Position sizing.

Two things live here: the correct Kelly fraction for a binary payoff, and an
explicit refusal to implement martingale.

Kelly for a binary option paying `b` on a win with win probability `w`:

    f* = (w(1 + b) - 1) / b

which is just `expected_value(w) / b`. At w=0.55, b=0.92 that is 6.1% of
bankroll. Full Kelly is the growth-optimal bet *given a known w*, and w is never
known — it is estimated, with error bars. Kelly is violently sensitive to
overestimating it: betting full Kelly on a w that is 2pp optimistic produces
negative expected growth. Hence `fraction=0.25` by default. Quarter Kelly gives
up about 6% of theoretical growth in exchange for a drastically smaller
drawdown and tolerance for estimation error, which is the right trade when your
edge estimate comes from a finite sample.

On martingale: with a 92% payout, doubling after a loss does not recover it. A
win returns 0.92x the stake against a 1.0x loss, so the recovery multiplier is
1/0.92 ~= 2.17x, and it compounds faster than the classic double-up. Starting at
2% of bankroll, ten consecutive losses require 0.02 * 2.17^10 ~= 48x bankroll.
At a 52% loss rate a 10-loss streak has probability 0.52^10 ~= 0.14%, so across
1,000 trades you expect to hit one roughly 1.4 times. Martingale converts a slow
negative edge into a fast total loss; it is not implemented here, and
`MartingaleSizing` exists only to raise with that explanation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .breakers import BreakerState


def break_even_rate(payout: float) -> float:
    """Win rate at which a binary contract has zero expected value."""
    if payout <= 0:
        raise ValueError("payout must be positive")
    return 1.0 / (1.0 + payout)


def kelly_fraction(win_rate: float, payout: float) -> float:
    """Full-Kelly bankroll fraction. Zero or negative means: do not bet.

    A negative result is not an instruction to bet the other way — the direction
    is part of the strategy, and a negative Kelly here means the strategy as
    specified is losing.
    """
    if payout <= 0:
        raise ValueError("payout must be positive")
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError("win_rate must be in [0, 1]")
    return (win_rate * (1.0 + payout) - 1.0) / payout


@dataclass
class SizingPolicy:
    """Fractional-Kelly sizing with hard caps.

    Attributes:
        win_rate: Estimated win rate. Use the *lower bound* of your out-of-sample
            confidence interval, not the point estimate — sizing off the point
            estimate systematically over-bets, because the cases where you
            overestimated are exactly the cases where you bet most.
        payout: Contract payout fraction.
        fraction: Kelly multiplier. 0.25 = quarter Kelly.
        max_fraction: Absolute cap on bankroll per trade, applied after Kelly.
        min_stake: Broker minimum; a computed stake below this trades nothing
            rather than rounding up into an oversized bet.
        flat_stake: If set, overrides Kelly with a fixed fraction of *starting*
            equity. Useful for evaluating a strategy without sizing effects
            confounding the win-rate measurement.
    """

    win_rate: float = 0.0
    payout: float = 0.92
    fraction: float = 0.25
    max_fraction: float = 0.02
    min_stake: float = 1.0
    flat_stake: float | None = None

    def target_fraction(self) -> float:
        """Bankroll fraction to risk per trade, after Kelly scaling and caps."""
        if self.flat_stake is not None:
            return max(0.0, self.flat_stake)
        k = kelly_fraction(self.win_rate, self.payout)
        if k <= 0:
            return 0.0
        return min(k * self.fraction, self.max_fraction)

    def stake(self, equity: float, state: BreakerState | None = None) -> float:
        """Stake for the next trade, or 0.0 to skip it."""
        if equity <= 0:
            return 0.0
        f = self.target_fraction()
        if f <= 0:
            return 0.0
        raw = equity * f
        if raw < self.min_stake:
            return 0.0
        return min(raw, equity)


class MartingaleSizing:
    """Not implemented, deliberately."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Martingale is not implemented. At a 92% payout the recovery "
            "multiplier is 1/0.92 = 2.17x per loss, so a 10-loss streak from a "
            "2% base needs ~48x bankroll — and at a 52% loss rate you expect "
            "~1.4 such streaks per 1,000 trades. It converts a negative edge "
            "into faster ruin, never into profit. Use SizingPolicy."
        )
