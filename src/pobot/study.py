"""End-to-end strategy study.

This is the module that actually answers the question. Everything else is
plumbing for it.

    features -> labels -> purged walk-forward CV -> out-of-sample predictions -> gate

Four properties make the answer trustworthy, and each one is a place the usual
retail backtest goes wrong:

1. **The model never sees its own test data.** Fitting *and* feature
   standardisation happen inside each training fold. Standardising over the full
   dataset first is the most common leak that survives cross-validation, because
   it hides in preprocessing rather than in the labels.

2. **The decision threshold is chosen on training data.** Picking the confidence
   margin that maximises *test* profit is just fitting the test set with extra
   steps. Here each fold picks its margin from its own training rows and lives
   with the result.

3. **Overlapping labels are purged.** A 60-second contract opened every few
   seconds shares almost all its price path with its neighbours, so unpurged
   folds train on samples that partly contain the test answers.

4. **The result is compared against the right baseline.** Beating 50% is not an
   edge. In a drifting market "always call" beats 50%, so the study reports the
   majority-direction win rate alongside the model's, and the model has to beat
   *that* as well as break-even.

The expected outcome is a fail. That is the point: a fail costs a few minutes of
compute, and the alternative way of discovering the same fact costs the account.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

from .backtest.contract import CALL, PUT, BinaryContract
from .backtest.labeler import label
from .data.store import TickSeries
from .features.core import DEFAULT_WINDOWS, build_features
from .model.logistic import LogisticSignal
from .validation.gate import GateResult, evaluate
from .validation.purged_cv import PurgedWalkForward, overlap_fraction

DEFAULT_MARGINS: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08)


class SignalModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SignalModel": ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class FoldReport:
    fold: int
    n_train: int
    n_test: int
    n_purged: int
    margin: float
    trades: int
    wins: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else float("nan")


@dataclass
class StudyResult:
    contract: BinaryContract
    folds: list[FoldReport]
    gate: GateResult
    trades: int
    wins: int
    base_rate: float
    majority_win_rate: float
    label_overlap: float
    n_samples: int
    notes: list[str] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else float("nan")

    @property
    def beats_majority(self) -> bool:
        return self.trades > 0 and self.win_rate > self.majority_win_rate

    @property
    def passed(self) -> bool:
        """A study passes only if it clears the gate *and* the trivial baseline."""
        return self.gate.passed and self.beats_majority

    def decay_note(self) -> Optional[str]:
        """Flag an edge that fades across time — a regime, not a signal."""
        rates = [f.win_rate for f in self.folds if f.trades > 0]
        if len(rates) < 4:
            return None
        half = len(rates) // 2
        early, late = float(np.mean(rates[:half])), float(np.mean(rates[half:]))
        if early - late > 0.02:
            return (
                f"win rate decays across folds: {early:.2%} early vs {late:.2%} "
                f"late — consistent with a regime that has ended, not a stable edge"
            )
        return None

    def report(self) -> str:
        be = self.contract.break_even_rate
        lines = [
            f"{'PASS' if self.passed else 'FAIL'}  out-of-sample study",
            "",
            f"  samples          {self.n_samples} labelled, "
            f"{self.label_overlap:.1%} mean label overlap",
            f"  trades taken     {self.trades}",
            f"  win rate         {self.win_rate:.2%}" if self.trades else
            "  win rate         n/a (no trades taken)",
            f"  break-even       {be:.2%}",
            f"  majority baseline{self.majority_win_rate:>7.2%}  "
            f"(always-{'call' if self.base_rate >= 0.5 else 'put'})",
            "",
            "  per fold:",
        ]
        for f in self.folds:
            wr = f"{f.win_rate:.2%}" if f.trades else "   n/a"
            lines.append(
                f"    fold {f.fold}: train={f.n_train:>6} test={f.n_test:>6} "
                f"purged={f.n_purged:>4} margin={f.margin:.3f} "
                f"trades={f.trades:>5} wr={wr}"
            )
        lines += ["", "  gate:", "    " + self.gate.report().replace("\n", "\n    ")]

        if self.trades and not self.beats_majority:
            lines.append("")
            lines.append(
                f"    - does not beat the majority baseline "
                f"({self.win_rate:.2%} vs {self.majority_win_rate:.2%}): the model "
                f"adds nothing over always trading one direction"
            )
        if (d := self.decay_note()):
            lines.append(f"    - {d}")
        for n in self.notes:
            lines.append(f"    - {n}")
        return "\n".join(lines)


def _pick_margin(
    proba: np.ndarray, y: np.ndarray, payout: float, margins: tuple[float, ...],
    min_trades: int,
) -> float:
    """Choose the confidence margin maximising expected profit on training rows.

    Total expected profit rather than EV per trade: a margin so tight it takes
    three trades can post a spectacular rate on nothing, and picking it would be
    fitting noise. Returns the smallest margin when none qualifies, so the fold
    still produces a measurable result instead of silently vanishing.
    """
    best_margin, best_profit = margins[0], -np.inf
    for m in margins:
        take_call = proba > 0.5 + m
        take_put = proba < 0.5 - m
        n = int(take_call.sum() + take_put.sum())
        if n < min_trades:
            continue
        wins = int(y[take_call].sum() + (1 - y[take_put]).sum())
        wr = wins / n
        profit = (wr * (1 + payout) - 1) * n
        if profit > best_profit:
            best_margin, best_profit = m, profit
    return best_margin


def run_study(
    series: TickSeries,
    contract: BinaryContract,
    *,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    n_splits: int = 5,
    margins: tuple[float, ...] = DEFAULT_MARGINS,
    l2: float = 1.0,
    alpha: float = 0.05,
    n_trials: int = 1,
    min_train: int = 200,
    min_fold_trades: int = 20,
    embargo_pct: float = 0.01,
    min_gate_trades: int = 1000,
    model_factory: Optional[Callable[[], SignalModel]] = None,
    stride: int = 1,
) -> StudyResult:
    """Run the full pipeline and return a gated out-of-sample verdict.

    Args:
        n_trials: Outer configurations you tried — window sets, model families,
            symbols, payout assumptions. The in-fold margin grid is *not*
            counted, because it is selected on training data and never sees the
            test rows. Everything you tried at this level does count, including
            what you discarded.
        stride: Take every `stride`-th tick as a candidate signal. Raising it
            cuts label overlap and compute at the cost of sample size; it does
            not change the statistics, since the overlap discount already prices
            in what densely-sampled signals are worth.
    """
    if stride < 1:
        raise ValueError("stride must be at least 1")

    notes: list[str] = []
    feats = build_features(series, windows)
    if stride > 1:
        feats = feats.subset(np.arange(0, len(feats), stride))
    if len(feats) < min_train * 2:
        raise ValueError(
            f"only {len(feats)} feature rows; need at least {min_train * 2}. "
            "Capture more data or shorten the feature windows."
        )

    # Label every candidate as a CALL. y = 1 means the price rose, so a call
    # would have won and a put would have lost. One label serves both directions.
    signal_ts = series.ts[feats.index]
    ls = label(series, signal_ts, np.full(len(signal_ts), CALL, dtype=np.int64), contract)
    if len(ls) == 0:
        raise ValueError("no contract could be settled — the series is too short")

    X = feats.matrix[ls.source_idx]
    decided = ~ls.ties
    n_ties = int((~decided).sum())
    if n_ties:
        notes.append(
            f"{n_ties} tied contracts excluded ({n_ties / len(ls):.1%}) — "
            "a high share suggests a coarsely quantised or inactive series"
        )
    X, y = X[decided], ls.win[decided].astype(np.float64)
    t0, t1 = ls.entry_ts[decided], ls.expiry_ts[decided]

    if len(X) < min_train * 2:
        raise ValueError(f"only {len(X)} decided contracts after dropping ties")

    base_rate = float(y.mean())
    majority = max(base_rate, 1 - base_rate)

    factory = model_factory or (lambda: LogisticSignal(l2=l2))
    cv = PurgedWalkForward(n_splits=n_splits, embargo_pct=embargo_pct, min_train=min_train)

    folds: list[FoldReport] = []
    traded_t0: list[np.ndarray] = []
    traded_t1: list[np.ndarray] = []
    total_trades = total_wins = 0

    for k, fold in enumerate(cv.split(t0, t1)):
        tr, te = fold.train_idx, fold.test_idx
        # A fold with one class present cannot teach direction; skip rather than
        # fit a model that will predict a constant and call it a signal.
        if len(np.unique(y[tr])) < 2:
            notes.append(f"fold {k} skipped: training labels are single-class")
            continue

        model = factory().fit(X[tr], y[tr])
        margin = _pick_margin(
            model.predict_proba(X[tr]), y[tr], contract.payout, margins, min_fold_trades
        )

        proba = model.predict_proba(X[te])
        take_call = proba > 0.5 + margin
        take_put = proba < 0.5 - margin
        taken = take_call | take_put

        wins = int(y[te][take_call].sum() + (1 - y[te][take_put]).sum())
        n_taken = int(taken.sum())

        total_trades += n_taken
        total_wins += wins
        if n_taken:
            traded_t0.append(t0[te][taken])
            traded_t1.append(t1[te][taken])

        folds.append(
            FoldReport(fold=k, n_train=len(tr), n_test=len(te), n_purged=fold.n_purged,
                       margin=margin, trades=n_taken, wins=wins)
        )

    if not folds:
        raise ValueError(
            "no usable folds — reduce n_splits or min_train, or capture more data"
        )

    if traded_t0:
        at0 = np.concatenate(traded_t0)
        at1 = np.concatenate(traded_t1)
        order = np.argsort(at0, kind="mergesort")
        overlap = overlap_fraction(at0[order], at1[order])
    else:
        overlap = 0.0
        notes.append("no trades taken: every fold's margin excluded all test rows")

    gate = evaluate(
        total_wins, total_trades, contract.payout,
        alpha=alpha, n_trials=n_trials, overlap=overlap, min_trades=min_gate_trades,
    )

    return StudyResult(
        contract=contract,
        folds=folds,
        gate=gate,
        trades=total_trades,
        wins=total_wins,
        base_rate=base_rate,
        majority_win_rate=majority,
        label_overlap=overlap,
        n_samples=len(X),
        notes=notes,
    )
