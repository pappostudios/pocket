"""Command line interface.

    pobot selftest     validate the pipeline against known-answer processes
    pobot capture      record broker + reference feeds to Parquet
    pobot summary      coverage report on captured data
    pobot power        sample size needed to detect a given edge
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .backtest.contract import CALL, PUT, BinaryContract
from .backtest.engine import run_backtest
from .data.store import capture_summary, ticks_to_series
from .feeds.synthetic import MeanRevertingFeed, RandomWalkFeed
from .risk.breakers import CircuitBreakers
from .risk.sizing import SizingPolicy
from .validation.gate import evaluate, required_sample_size
from .validation.purged_cv import PurgedWalkForward, overlap_fraction

#: Risk limits protect capital; they have no place in an edge measurement. Left
#: on, a loss-streak breaker truncates the sample right after a losing run and
#: biases the measured win rate upward. Off for research, on for trading.
_MEASURE_ONLY = CircuitBreakers(
    max_daily_loss=1.0, max_drawdown=1.0,
    max_consecutive_losses=10**9, max_trades_per_day=10**9,
)


def _momentum(view):
    """Trivial momentum rule, used only to exercise the machinery."""
    h = view.history(10)
    if len(h) < 10:
        return None
    return CALL if h[-1] > h[0] else PUT


def _reversion(view):
    """Fade deviation from a short mean — the rule an OU process rewards."""
    h = view.history(20)
    if len(h) < 20:
        return None
    return PUT if h[-1] > h.mean() else CALL


def cmd_selftest(args: argparse.Namespace) -> int:
    """Known-answer tests. Trust nothing this repo says until these pass."""
    contract = BinaryContract(payout=0.92, duration_s=60, entry_latency_ms=250)
    sizing = SizingPolicy(flat_stake=0.01)
    n_ticks = args.ticks

    print("=" * 68)
    print("SELFTEST 1: random walk — no edge exists, none should be found")
    print("=" * 68)
    rw = RandomWalkFeed(["EURUSD"], interval_ms=1000, seed=args.seed)
    series = ticks_to_series(rw.generate(n_ticks), "EURUSD", rw.name)
    res = run_backtest(series, _momentum, contract, sizing=sizing,
                       breakers=_MEASURE_ONLY, warmup=20)

    s = res.summary()
    print(f"trades={s['trades']} win_rate={s['win_rate']:.2%} "
          f"break_even={s['break_even_rate']:.2%} return={s['return_pct']:+.1f}%")

    wins = int(res.blotter["win"].sum()) if res.n_trades else 0
    decided = int((res.blotter["ret"] != 0).sum()) if res.n_trades else 0
    ov = (
        overlap_fraction(res.blotter["entry_ts"].to_numpy(), res.blotter["expiry_ts"].to_numpy())
        if res.n_trades else 0.0
    )
    gate = evaluate(wins, decided, contract.payout, overlap=ov, min_trades=args.min_trades)
    print(gate.report())

    if gate.passed:
        print("\n!! FAILURE: an edge was reported on a driftless random walk.")
        print("!! No such edge can exist. This is a bug — most likely lookahead")
        print("!! in the labeller or the engine. Do not trust any other result.")
        return 1
    print("\nOK: no edge found where none exists.")

    print()
    print("=" * 68)
    print("SELFTEST 2: mean-reverting process — a planted edge must be detected")
    print("=" * 68)
    ou = MeanRevertingFeed(["EURUSD"], kappa=0.30, interval_ms=1000, seed=args.seed)
    series2 = ticks_to_series(ou.generate(n_ticks), "EURUSD", ou.name)
    res2 = run_backtest(series2, _reversion, contract, sizing=sizing,
                        breakers=_MEASURE_ONLY, warmup=20)
    s2 = res2.summary()
    print(f"trades={s2['trades']} win_rate={s2['win_rate']:.2%} "
          f"break_even={s2['break_even_rate']:.2%} return={s2['return_pct']:+.1f}%")

    if res2.n_trades and s2["win_rate"] > s2["break_even_rate"]:
        print("\nOK: planted edge detected. The pipeline can find a real signal.")
    else:
        print("\n!! The pipeline failed to detect a deliberately planted edge.")
        print("!! It will not detect a real one either.")
        return 1

    print()
    print("=" * 68)
    print("SELFTEST 3: purged walk-forward removes overlapping labels")
    print("=" * 68)
    # Deliberately allow concurrent positions here. Serial trades never overlap,
    # so they would show purged=0 and prove nothing. Overlapping labels are the
    # condition purging exists to handle, and the condition any signal firing
    # more often than once per expiry actually produces.
    res3 = run_backtest(series2, _reversion, contract, sizing=sizing,
                        breakers=_MEASURE_ONLY, warmup=20, allow_concurrent=True)
    t0 = res3.blotter["entry_ts"].to_numpy()
    t1 = res3.blotter["expiry_ts"].to_numpy()
    ov3 = overlap_fraction(t0, t1)
    print(f"  {len(t0)} overlapping trades, mean label overlap {ov3:.1%}")

    total_purged = 0
    for k, fold in enumerate(PurgedWalkForward(n_splits=4, min_train=5).split(t0, t1)):
        leak = int((t1[fold.train_idx] >= fold.test_start_ms).sum())
        total_purged += fold.n_purged
        status = "clean" if leak == 0 else f"LEAK x{leak}"
        print(f"  fold {k}: train={len(fold.train_idx):>5} test={len(fold.test_idx):>5} "
              f"purged={fold.n_purged:>3} -> {status}")
        if leak:
            print("\n!! Training data leaked into a test window. Every out-of-sample")
            print("!! number this pipeline produces would be optimistic.")
            return 1
    if total_purged == 0:
        print("\n!! Purging removed nothing despite overlapping labels — the purge")
        print("!! step is not working.")
        return 1
    print(f"\nOK: {total_purged} contaminated training samples removed, no leakage.")

    print()
    print("=" * 68)
    print("Sample sizes needed at a 92% payout (break-even 52.08%)")
    print("=" * 68)
    for wr in (0.53, 0.54, 0.55, 0.56):
        print(f"  true {wr:.0%} -> {required_sample_size(contract.break_even_rate, wr):>7,} trades")
    print("\nAll selftests passed.")
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    from .capture.recorder import Recorder, RecorderConfig
    from .feeds.pocketoption import PocketOptionFeed
    from .feeds.reference import ReferenceFeed
    from .feeds.wsfeed import ProtocolNotConfigured

    symbols = args.symbols.split(",")
    feeds = [PocketOptionFeed(symbols)]
    if not args.no_reference:
        feeds.append(ReferenceFeed(symbols))

    rec = Recorder(feeds, RecorderConfig(out_dir=Path(args.out)))
    failures: dict[str, BaseException] = {}
    try:
        failures = asyncio.run(rec.run())
    except KeyboardInterrupt:
        print("\nstopped; buffers flushed")

    if not failures:
        return 0

    print(file=sys.stderr)
    for name, exc in failures.items():
        print(f"feed {name} failed: {exc}", file=sys.stderr)
    if any(isinstance(e, ProtocolNotConfigured) for e in failures.values()):
        print("\nDerive the wire format from live traffic first — see "
              "docs/PROTOCOL.md.", file=sys.stderr)

    if len(failures) == len(feeds):
        print("\nNothing was captured.", file=sys.stderr)
        return 2

    # A partial capture is still worth keeping, but losing the reference feed
    # costs the lag and synthetic-series analyses, so it must not exit clean.
    print(f"\nPartial capture: {rec.rows_written} rows from "
          f"{len(feeds) - len(failures)} of {len(feeds)} feeds.", file=sys.stderr)
    return 1


def cmd_summary(args: argparse.Namespace) -> int:
    df = capture_summary(Path(args.dir))
    if df.empty:
        print("no captured data found")
        return 1
    print(df.to_string(index=False))
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    be = 1.0 / (1.0 + args.payout)
    print(f"payout {args.payout:.0%} -> break-even win rate {be:.2%}")
    print(f"coin-flip EV per trade: {0.5 * (1 + args.payout) - 1:+.2%}\n")
    print(f"{'true win rate':>14} {'edge (pp)':>10} {'trades needed':>14}")
    for wr in args.rates:
        if wr <= be:
            print(f"{wr:>13.1%} {(wr - be) * 100:>10.2f} {'never — negative EV':>14}")
            continue
        n = required_sample_size(be, wr, alpha=args.alpha, power=args.power)
        print(f"{wr:>13.1%} {(wr - be) * 100:>10.2f} {n:>14,}")
    print(f"\n(one-sided, alpha={args.alpha}, power={args.power})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pobot", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest", help="validate the pipeline on known-answer data")
    st.add_argument("--ticks", type=int, default=60000)
    st.add_argument("--seed", type=int, default=7)
    st.add_argument("--min-trades", type=int, default=500)
    st.set_defaults(func=cmd_selftest)

    cap = sub.add_parser("capture", help="record broker + reference feeds")
    cap.add_argument("--symbols", default="EURUSD")
    cap.add_argument("--out", default="data/ticks")
    cap.add_argument("--no-reference", action="store_true",
                     help="record the broker feed only (disables lag analysis)")
    cap.set_defaults(func=cmd_capture)

    summ = sub.add_parser("summary", help="coverage report on captured data")
    summ.add_argument("--dir", default="data/ticks")
    summ.set_defaults(func=cmd_summary)

    pw = sub.add_parser("power", help="sample size needed to detect an edge")
    pw.add_argument("--payout", type=float, default=0.92)
    pw.add_argument("--alpha", type=float, default=0.05)
    pw.add_argument("--power", type=float, default=0.80)
    pw.add_argument("--rates", type=float, nargs="+",
                    default=[0.53, 0.54, 0.55, 0.56, 0.58, 0.60])
    pw.set_defaults(func=cmd_power)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
