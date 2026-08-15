"""Binary option contract mechanics and the arithmetic that governs everything.

The central fact this module encodes: a binary option pays `payout` on a win and
takes the entire stake on a loss. Break-even win rate is therefore

    w* = 1 / (1 + payout)

which at a 92% payout is 52.08%, at 85% is 54.05%, and at 80% is 55.56%. A
coin-flip strategy at 92% payout returns 0.5 * 1.92 - 1 = -4% per trade — a
worse per-bet edge than single-zero roulette, compounded at whatever rate the
bot fires.

Every other module exists to answer one question: is the realised win rate
*reliably* above w*, out of sample, after costs. Nothing else about a strategy
matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..feeds.base import Millis

Direction = Literal[-1, 1]  # +1 = "call"/higher, -1 = "put"/lower

CALL: Direction = 1
PUT: Direction = -1


@dataclass(frozen=True)
class BinaryContract:
    """Terms of a single binary trade.

    Attributes:
        payout: Fraction returned on a win, on top of the stake (0.92 = 92%).
        duration_s: Contract life in seconds.
        entry_latency_ms: Delay between the bot deciding and the strike being
            fixed — signal computation, network transit, broker acceptance. Set
            it from *measured* round-trips, not from hope. At 60-second
            expiries a few hundred milliseconds of unmodelled latency is enough
            to move a marginal strategy across the break-even line, so an
            optimistic value here manufactures an edge that will not survive
            live trading.
        expiry_mode: "relative" expires at entry + duration. "clock" expires at
            the next wall-clock boundary, which is how most platforms implement
            short expiries. Under "clock" the true contract life varies from
            near-zero to a full period, and a strategy tuned under "relative"
            can behave completely differently — match the platform.
        tie_rule: Settlement when the expiry price exactly equals the strike.
            "refund" returns the stake, "loss" forfeits it. Ties are rare on
            continuous FX but common on coarsely quantised or low-activity
            synthetic assets, where an unfavourable tie rule is a real cost.
        spread: Optional half-spread charged against the strike, in price units.
            Applied adversely: the strike moves against the trade's direction.
    """

    payout: float = 0.92
    duration_s: int = 60
    entry_latency_ms: int = 250
    expiry_mode: Literal["relative", "clock"] = "relative"
    tie_rule: Literal["refund", "loss"] = "refund"
    spread: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.payout < 5.0:
            raise ValueError(f"implausible payout: {self.payout} (expected e.g. 0.92)")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.entry_latency_ms < 0:
            raise ValueError("entry_latency_ms must be non-negative")
        if self.spread < 0:
            raise ValueError("spread must be non-negative")

    @property
    def break_even_rate(self) -> float:
        """Win rate at which expected value is exactly zero."""
        return 1.0 / (1.0 + self.payout)

    def expected_value(self, win_rate: float) -> float:
        """Expected return per unit stake at a given win rate.

        Negative for any win rate below `break_even_rate`. This function is the
        reason the statistical gate is strict: at a 92% payout, being wrong
        about your win rate by two percentage points flips the sign.
        """
        return win_rate * (1.0 + self.payout) - 1.0

    def entry_ts(self, signal_ts: Millis) -> Millis:
        """When the strike is actually fixed, given a decision at `signal_ts`."""
        return signal_ts + self.entry_latency_ms

    def expiry_ts(self, entry_ts: Millis) -> Millis:
        """Settlement time for a contract entered at `entry_ts`."""
        dur_ms = self.duration_s * 1000
        if self.expiry_mode == "relative":
            return entry_ts + dur_ms
        # Clock mode: next boundary strictly after entry. An entry landing
        # exactly on a boundary gets the following one, never a zero-life
        # contract that would settle at its own strike.
        return ((entry_ts // dur_ms) + 1) * dur_ms

    def effective_strike(self, raw_strike: float, direction: Direction) -> float:
        """Strike after adverse spread.

        A call is filled above the mid and a put below it, so the spread always
        works against the position. Modelling it favourably — or not at all when
        the broker charges one — is a quiet way to invent an edge.
        """
        return raw_strike + direction * self.spread

    def settle(
        self, strike: float, expiry_price: float, direction: Direction
    ) -> float:
        """Net return per unit stake: +payout on a win, -1 on a loss.

        Under `tie_rule="refund"` an exact tie returns 0.0 (stake back).
        """
        if expiry_price == strike:
            return 0.0 if self.tie_rule == "refund" else -1.0
        moved_up = expiry_price > strike
        won = moved_up if direction == CALL else not moved_up
        return self.payout if won else -1.0


# Payout tiers seen across typical retail binary platforms, with the win rate
# each requires. Useful as a reality check when a backtest reports, say, 53%:
# that is profitable at 92% and loss-making at everything below it.
PAYOUT_TABLE = {
    0.92: 0.5208,
    0.85: 0.5405,
    0.80: 0.5556,
    0.70: 0.5882,
}
