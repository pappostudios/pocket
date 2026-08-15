"""Command line interface.

    pobot selftest     validate the pipeline against known-answer processes
    pobot capture      record broker + reference feeds to Parquet
    pobot summary      coverage report on captured data
    pobot power        sample size needed to detect a given edge
    pobot fingerprint  test a series against the random-walk null
    pobot lag          estimate how far the broker feed trails a reference feed
    pobot study        full out-of-sample study, gated

The three analysis commands accept `--demo rw|ou` in place of captured data, so
the pipeline can be exercised end to end before any capture exists. `rw` is a
driftless random walk (no edge can exist); `ou` is mean-reverting (an edge is
planted). Run both — a tool that only ever says "no" is not a test.
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
    print("SELFTEST 4: full study — the decisive end-to-end check")
    print("=" * 68)
    # Everything above tests a component. This runs the whole pipeline the way a
    # real evaluation would: features, purged CV, in-fold threshold selection,
    # then the gate. It must fail on the random walk and pass on the OU process.
    from .study import run_study

    rw_study = run_study(series, contract, n_splits=5, stride=5, min_gate_trades=200)
    print(f"  random walk:    win rate {rw_study.win_rate:.2%} over "
          f"{rw_study.trades} trades -> "
          f"{'PASS' if rw_study.passed else 'FAIL'} (expected FAIL)")
    if rw_study.passed:
        print("\n!! The study reported a tradeable edge on a driftless random walk.")
        print("!! No such edge exists. Every result this pipeline produces is")
        print("!! unreliable until this is fixed.")
        return 1

    ou_study = run_study(series2, contract, n_splits=5, stride=5, min_gate_trades=200)
    print(f"  mean reverting: win rate {ou_study.win_rate:.2%} over "
          f"{ou_study.trades} trades -> "
          f"{'PASS' if ou_study.passed else 'FAIL'} (expected PASS)")
    if not ou_study.passed:
        print("\n!! The study failed to find a deliberately planted edge. It will")
        print("!! not find a real one either.")
        return 1
    print("\nOK: the study rejects noise and detects a real signal.")

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


def _demo_series(kind: str, n: int, seed: int = 7):
    """Known-answer series for exercising a command without captured data."""
    if kind == "rw":
        feed = RandomWalkFeed(["DEMO"], interval_ms=1000, seed=seed)
    elif kind == "ou":
        feed = MeanRevertingFeed(["DEMO"], kappa=0.30, interval_ms=1000, seed=seed)
    else:
        raise ValueError(f"unknown demo series {kind!r} (expected 'rw' or 'ou')")
    return ticks_to_series(feed.generate(n), "DEMO", feed.name)


def _resolve_series(args: argparse.Namespace, *, source: str, n: int = 40_000):
    """Load a captured series, or synthesise one in demo mode."""
    if getattr(args, "demo", None):
        print(f"[demo: {args.demo}] synthetic series, no captured data used\n")
        return _demo_series(args.demo, n)

    from .data.store import load_ticks, to_series

    df = load_ticks(Path(args.dir), symbol=args.symbol, source=source)
    if df.empty:
        raise SystemExit(
            f"no ticks for symbol={args.symbol} source={source} under {args.dir}. "
            "Run `pobot summary` to see what was captured, or pass --demo rw|ou."
        )
    return to_series(df, args.symbol, source)


def cmd_fingerprint(args: argparse.Namespace) -> int:
    from .analysis.fingerprint import fingerprint

    series = _resolve_series(args, source=args.source)
    print(f"symbol={series.symbol} source={series.source}\n")
    print(fingerprint(series, alpha=args.alpha).report())
    return 0


def cmd_lag(args: argparse.Namespace) -> int:
    from .analysis.lag import estimate_lag

    if args.demo:
        # Broker feed is an exact 300ms-delayed copy — the detectable case.
        from .feeds.synthetic import LaggedFeed

        base = RandomWalkFeed(["DEMO"], interval_ms=100, seed=9)
        reference = ticks_to_series(base.generate(8000), "DEMO", base.name)
        lagged = LaggedFeed(RandomWalkFeed(["DEMO"], interval_ms=100, seed=9), 300)
        broker = ticks_to_series(lagged.generate(8000), "DEMO", lagged.name)
        print("[demo] broker feed is a 300ms-delayed copy of the reference\n")
    else:
        broker = _resolve_series(args, source=args.broker_source)
        reference = _resolve_series(args, source=args.reference_source)

    print(estimate_lag(broker, reference, grid_ms=args.grid_ms,
                       max_lag_ms=args.max_lag_ms, alpha=args.alpha).report())
    return 0


def cmd_study(args: argparse.Namespace) -> int:
    from .study import run_study

    contract = BinaryContract(
        payout=args.payout, duration_s=args.duration,
        entry_latency_ms=args.latency, spread=args.spread,
    )
    series = _resolve_series(args, source=args.source)

    print(f"contract: payout={contract.payout:.0%} duration={contract.duration_s}s "
          f"latency={contract.entry_latency_ms}ms -> break-even "
          f"{contract.break_even_rate:.2%}\n")

    result = run_study(
        series, contract, n_splits=args.splits, stride=args.stride,
        n_trials=args.trials, alpha=args.alpha, min_gate_trades=args.min_trades,
    )
    print(result.report())
    return 0 if result.passed else 1


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

    fp = sub.add_parser("fingerprint", help="test a series against the random-walk null")
    fp.add_argument("--dir", default="data/ticks")
    fp.add_argument("--symbol", default="EURUSD")
    fp.add_argument("--source", default="pocketoption")
    fp.add_argument("--alpha", type=float, default=0.05)
    fp.add_argument("--demo", choices=["rw", "ou"], help="use a known-answer series")
    fp.set_defaults(func=cmd_fingerprint)

    lg = sub.add_parser("lag", help="estimate how far the broker feed trails a reference")
    lg.add_argument("--dir", default="data/ticks")
    lg.add_argument("--symbol", default="EURUSD")
    lg.add_argument("--broker-source", default="pocketoption")
    lg.add_argument("--reference-source", default="reference")
    lg.add_argument("--grid-ms", type=int, default=100)
    lg.add_argument("--max-lag-ms", type=int, default=3000)
    lg.add_argument("--alpha", type=float, default=0.05)
    lg.add_argument("--demo", action="store_const", const="rw",
                    help="use a synthetic 300ms-delayed feed")
    lg.set_defaults(func=cmd_lag)

    sd = sub.add_parser("study", help="full out-of-sample study, gated")
    sd.add_argument("--dir", default="data/ticks")
    sd.add_argument("--symbol", default="EURUSD")
    sd.add_argument("--source", default="pocketoption")
    sd.add_argument("--payout", type=float, default=0.92)
    sd.add_argument("--duration", type=int, default=60)
    sd.add_argument("--latency", type=int, default=250,
                    help="measured signal-to-fill latency in ms; guessing low invents an edge")
    sd.add_argument("--spread", type=float, default=0.0)
    sd.add_argument("--splits", type=int, default=5)
    sd.add_argument("--stride", type=int, default=5)
    sd.add_argument("--alpha", type=float, default=0.05)
    sd.add_argument("--min-trades", type=int, default=1000)
    sd.add_argument("--trials", type=int, default=1,
                    help="outer configurations tried, INCLUDING the ones you discarded")
    sd.add_argument("--demo", choices=["rw", "ou"], help="use a known-answer series")
    sd.set_defaults(func=cmd_study)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
