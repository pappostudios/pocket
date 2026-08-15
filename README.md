# pobot

A research pipeline for deciding whether a binary-options strategy has an edge —
**before** any money is involved.

This is deliberately not a trading bot. There is no order-placement code in it.
Placing orders is the last step, and almost nobody reaches it honestly.

## The arithmetic

A binary option pays `p` on a win and takes the entire stake on a loss, so it
breaks even at a win rate of `1 / (1 + p)`:

| Payout | Break-even win rate |
|-------:|--------------------:|
| 92% | 52.08% |
| 85% | 54.05% |
| 80% | 55.56% |
| 70% | 58.82% |

A coin flip at a 92% payout returns `0.5 × 1.92 − 1 = −4%` per trade — a worse
per-bet edge than single-zero roulette, compounded at whatever rate a bot fires.
Flat-staking 2% of bankroll across 100 trades a day with no edge burns roughly
8% of bankroll per day in expectation.

So there is exactly one question worth engineering around: **is the win rate
reliably above break-even, out of sample, after costs?** Everything here serves
that question.

## What it costs to answer

At a 92% payout, a one-sided test at 95% confidence and 80% power needs:

| True win rate | Edge over break-even | Trades needed |
|--------------:|---------------------:|--------------:|
| 53% | 0.9 pp | 18,351 |
| 54% | 1.9 pp | 4,194 |
| 55% | 2.9 pp | 1,809 |
| 56% | 3.9 pp | 1,002 |

Required sample scales with the inverse square of the gap to break-even, which
is why "it won 7 of the last 10" is not evidence of anything. A strategy showing
58% over 200 trades has a 95% interval spanning roughly 51%–65% — consistent
with a strong edge and consistent with nothing.

```
pobot power --payout 0.92     # run the table for your own payout
```

## Quick start

```bash
pip install -e ".[dev]"
pobot selftest      # validate the pipeline before trusting anything it says
pytest -q
```

`selftest` is the first thing to run and the first thing to re-run after any
change. It backtests on a driftless random walk — where no edge can exist — and
fails loudly if the pipeline reports one, because that means lookahead bias. It
then confirms the pipeline *does* detect a deliberately planted edge in a
mean-reverting series, since a test that can only say "no" is not a test.

## Pipeline

```
feeds/       broker + independent reference prices, spec-driven, pure transport
capture/     durable hour-partitioned Parquet, crash-safe via tmp→rename
data/        read back ordered by arrival time, not by broker-claimed time
backtest/    contract mechanics, honest labelling, lookahead-proof engine
validation/  purged walk-forward CV, then the statistical gate
risk/        fractional Kelly and circuit breakers
```

### Phase 1 — Capture

```bash
pobot capture --symbols EURUSD --out data/ticks
pobot summary --dir data/ticks
```

Record the broker feed **and** an independent reference feed simultaneously.
This is not optional detail. The broker's stream cannot be used to audit the
broker's stream, and two analyses depend on having both — neither of which can
be run retroactively on broker-only data:

- **Lag estimation.** Cross-correlate the two series at a range of offsets. A
  consistent nonzero peak means the broker feed is a delayed copy of the real
  market — the one structurally sound edge in this product, and also the one
  brokers explicitly act against.
- **Synthetic-series detection.** On "OTC" assets the broker quotes prices while
  the real market is closed. If the two series decouple, the broker's is
  generated rather than observed, and its generator may have exploitable
  structure.

Both feeds need a `ProtocolSpec` derived from live traffic — see
[docs/PROTOCOL.md](docs/PROTOCOL.md). Unconfigured feeds raise rather than
silently record nothing.

### Phase 2 — Backtest

The engine walks tick by tick and hands the strategy a `MarketView` that
**cannot** see past the current index; reading forward raises `LookaheadError`.
Lookahead becomes a crash rather than a suspiciously good Sharpe ratio.

```python
from pobot.backtest.contract import CALL, PUT, BinaryContract
from pobot.backtest.engine import run_backtest

contract = BinaryContract(payout=0.92, duration_s=60, entry_latency_ms=250)

def strategy(view):
    h = view.history(20)
    return None if len(h) < 20 else (PUT if h[-1] > h.mean() else CALL)

result = run_backtest(series, strategy, contract)
print(result.summary())
```

Set `entry_latency_ms` from *measured* round-trips. At 60-second expiries a few
hundred unmodelled milliseconds is enough to move a marginal strategy across the
break-even line.

### Phase 3 — Validate, then gate

```python
from pobot.validation.purged_cv import PurgedWalkForward, overlap_fraction
from pobot.validation.gate import evaluate
```

`PurgedWalkForward` is the highest-value component here. A trade opened at `t0`
settles at `t1`; two trades opened seconds apart resolve on nearly the same price
path and are close to one observation counted twice. Ordinary k-fold CV lets a
training sample overlap a test sample and carry its answer. Walk-forward ordering,
purging overlapping labels, and an embargo after each test window remove that.
This leak is *the* reason retail bots backtest profitably and lose live.

`evaluate` then decides, and it is strict on purpose:

- Requires a minimum effective sample — below it, no win rate is informative.
- Requires the **lower** confidence bound to beat break-even, not the point
  estimate. That is the difference between "probably profitable" and "profitable
  unless I was unlucky in a way the data cannot rule out".
- Applies a Šidák correction via `n_trials`. Grid-searching indicator parameters
  *is* testing hundreds of hypotheses; at α=0.05, about 5 in 100 look significant
  by chance alone. Set `n_trials` to every configuration you tried, including
  the ones you discarded.
- Discounts for label overlap via `overlap_fraction`.

**Failing is the expected outcome.** The correct response to a fail is to discard
the strategy, not to re-tune it on the same data until it passes — that is
exactly the multiple-testing problem the correction exists to price in.

## Kill criteria

Write these down before looking at results:

- Out-of-sample win rate not significantly above `1/(1+p)` → **discard.**
- Edge in-sample but not in walk-forward → **overfit, discard.**
- Edge decaying across time folds → **a regime, not a signal.**
- Edge that disappears when `entry_latency_ms` is set realistically → **it was
  never there.**

## Risk

`SizingPolicy` implements fractional Kelly. For a binary payoff:

```
f* = (w(1 + b) − 1) / b
```

At w=0.55, b=0.92 that is 6.1% of bankroll at full Kelly. Default is quarter
Kelly, because `w` is estimated rather than known and Kelly is violently
sensitive to overestimating it — betting full Kelly on a `w` that is 2pp
optimistic produces negative expected growth. Size off the *lower bound* of your
confidence interval, not the point estimate.

**Martingale is not implemented, and `MartingaleSizing` raises.** At a 92%
payout, doubling after a loss does not recover it: a win returns 0.92× against a
1.0× loss, so the recovery multiplier is `1/0.92 ≈ 2.17×`. From a 2% base, ten
consecutive losses need ~48× bankroll — and at a 52% loss rate you expect roughly
1.4 such streaks per 1,000 trades. It converts a slow negative edge into a fast
total loss.

`CircuitBreakers` enforces daily loss limits, drawdown halts, loss-streak pauses,
trade caps, and feed-staleness refusal. Drawdown and ruin latch permanently; day-
scoped limits pause until the UTC date rolls. Disable them for edge *measurement*
(they truncate the sample right after a losing run, biasing the win rate upward)
and never for live trading.

## What this does not do

No order placement, by design. Automated trading is restricted under Pocket
Option's terms, which reserve the right to void trades and close accounts.
Adding execution is a decision to make with the terms in front of you, and only
after phases 1–3 have actually passed.

Two practical notes if you get that far: paper-trade on the live feed through
the same code path first — a gap between paper and backtest means your fill
model is wrong — and verify a withdrawal works with a small amount before
scaling. The binding constraint on this kind of account is usually getting money
out, not the strategy.

## Honest expectations

You are looking for a 2-percentage-point edge against a counterparty that sets
the prices, sets the payouts, and can change both. The pipeline is built so that
a negative answer arrives cheaply and credibly, which is the most likely useful
outcome. Pointing the same architecture at a venue with real order books and no
fixed house cut starts from zero edge rather than −4%.
