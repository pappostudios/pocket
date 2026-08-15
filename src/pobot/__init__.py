"""pobot — a research pipeline for binary-options strategy evaluation.

Deliberately *not* a trading bot. There is no order-placement code here, because
placing orders is the last step and almost nobody gets to it honestly. What this
package does is answer the prior question: does a candidate strategy have an edge
that survives out-of-sample testing after the payout structure is priced in?

The arithmetic that motivates everything: a binary option paying `p` on a win and
taking the full stake on a loss breaks even at a win rate of 1/(1+p) — 52.08% at
a 92% payout, 55.56% at 80%. A coin flip at 92% returns -4% per trade. So the bar
is not "is this strategy profitable in a backtest", it is "is this strategy's win
rate reliably above 52%, out of sample, with enough trades to tell".

Pipeline:

    feeds/      capture broker + independent reference prices
    capture/    durable hour-partitioned Parquet recording
    data/       read it back, ordered by arrival time
    backtest/   contract mechanics, honest labelling, lookahead-proof engine
    validation/ purged walk-forward CV, then the statistical gate
    risk/       fractional Kelly and circuit breakers

Run `pobot selftest` first. It backtests on a driftless random walk, where no
edge can exist, and confirms the pipeline correctly reports nothing — the fastest
way to catch the lookahead bugs that make broken backtests look profitable.
"""

from .backtest.contract import CALL, PUT, BinaryContract
from .validation.gate import evaluate, required_sample_size

__version__ = "0.1.0"

__all__ = [
    "BinaryContract",
    "CALL",
    "PUT",
    "evaluate",
    "required_sample_size",
    "__version__",
]
