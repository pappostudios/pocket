import json

import pytest

from pobot.capture.writer import TickWriter
from pobot.data.store import capture_summary, load_ticks, to_series
from pobot.feeds.base import Tick
from pobot.feeds.pocketoption import PocketOptionFeed
from pobot.feeds.synthetic import LaggedFeed, RandomWalkFeed
from pobot.feeds.wsfeed import ProtocolNotConfigured, ProtocolSpec, SpecDrivenWebSocketFeed


def test_write_then_read_roundtrip(tmp_path):
    feed = RandomWalkFeed(["EURUSD"], interval_ms=1000, seed=3)
    ticks = feed.generate(500)
    with TickWriter(tmp_path, "test", buffer_rows=100) as w:
        for t in ticks:
            w.append(t)

    df = load_ticks(tmp_path)
    assert len(df) == len(ticks)
    assert set(df["symbol"]) == {"EURUSD"}


def test_loaded_ticks_are_ordered_by_arrival(tmp_path):
    """Ordering is by ts_recv, not ts_event: arrival order is what you knew, when."""
    with TickWriter(tmp_path, "s", buffer_rows=10) as w:
        for i, ts_event in enumerate([500, 300, 400, 100]):
            w.append(Tick("X", 1.0 + i, ts_event=ts_event, ts_recv=1000 + i, source="s"))
    df = load_ticks(tmp_path)
    assert list(df["ts_recv"]) == sorted(df["ts_recv"])


def test_existing_files_are_never_overwritten(tmp_path):
    feed = RandomWalkFeed(["X"], seed=1)
    ticks = feed.generate(50)
    for _ in range(2):
        with TickWriter(tmp_path, "s", buffer_rows=1000) as w:
            for t in ticks:
                w.append(t)
    assert len(load_ticks(tmp_path)) == 2 * len(ticks)


def test_no_tmp_files_survive_a_clean_close(tmp_path):
    with TickWriter(tmp_path, "s") as w:
        for t in RandomWalkFeed(["X"], seed=1).generate(10):
            w.append(t)
    assert list(tmp_path.rglob("*.tmp")) == []


def test_capture_summary_reports_coverage(tmp_path):
    with TickWriter(tmp_path, "s") as w:
        for t in RandomWalkFeed(["A", "B"], seed=1).generate(100):
            w.append(t)
    summary = capture_summary(tmp_path)
    assert set(summary["symbol"]) == {"A", "B"}
    assert (summary["ticks"] == 100).all()


def test_to_series_is_sorted_and_searchable(tmp_path):
    feed = RandomWalkFeed(["EURUSD"], interval_ms=1000, seed=5)
    with TickWriter(tmp_path, feed.name) as w:
        for t in feed.generate(200):
            w.append(t)
    s = to_series(load_ticks(tmp_path), "EURUSD", feed.name)
    assert len(s) == 200
    assert (s.ts[1:] >= s.ts[:-1]).all()
    assert s.price_at(int(s.ts[10])) == pytest.approx(float(s.price[10]))


# --- Feed protocol handling -------------------------------------------------

def test_unconfigured_feed_fails_loudly():
    """An empty spec must raise, never silently record nothing."""
    feed = PocketOptionFeed(["EURUSD"])
    with pytest.raises(ProtocolNotConfigured):
        import asyncio

        async def drain():
            async for _ in feed.stream():
                break

        asyncio.run(drain())


def test_spec_driven_decode_extracts_ticks():
    spec = ProtocolSpec(url="wss://x", strip_prefix="0123456789", quotes_path="1",
                        symbol_key="asset", price_key="price", ts_key="time",
                        ts_scale=1000.0)
    feed = SpecDrivenWebSocketFeed(["EURUSD"], spec, name="t")
    frame = '42["stream",[{"asset":"EURUSD","price":1.2345,"time":1700000000}]]'
    ticks = feed.decode(frame)
    assert len(ticks) == 1
    assert ticks[0].symbol == "EURUSD"
    assert ticks[0].price == pytest.approx(1.2345)
    assert ticks[0].ts_event == 1_700_000_000_000


def test_control_frames_decode_to_nothing_without_erroring():
    spec = ProtocolSpec(url="wss://x", quotes_path="1")
    feed = SpecDrivenWebSocketFeed(["X"], spec, name="t")
    for frame in ["2", "3probe", "", "not json"]:
        assert feed.decode(frame) == []
    assert feed.stats.errors == 0


def test_malformed_price_is_counted_not_crashed():
    spec = ProtocolSpec(url="wss://x", quotes_path="", symbol_key="s",
                        price_key="p", ts_key="")
    feed = SpecDrivenWebSocketFeed(["X"], spec, name="t")
    assert feed.decode(json.dumps({"s": "X", "p": "abc"})) == []
    assert feed.stats.errors == 1


# --- Lag hypothesis ---------------------------------------------------------

def test_lagged_feed_shifts_timestamps_only():
    """A delayed copy of a feed carries identical prices, later."""
    inner = RandomWalkFeed(["X"], interval_ms=100, seed=2)
    lagged = LaggedFeed(inner, lag_ms=200)
    base, late = inner.generate(50), lagged.generate(50)
    assert [t.price for t in base] == [t.price for t in late]
    assert all(b.ts_recv + 200 == l.ts_recv for b, l in zip(base, late))


# --- Recorder failure reporting --------------------------------------------

def test_recorder_reports_a_totally_failed_capture(tmp_path):
    """A capture that recorded nothing must never look like a success."""
    import asyncio

    from pobot.capture.recorder import Recorder, RecorderConfig

    rec = Recorder([PocketOptionFeed(["EURUSD"])], RecorderConfig(out_dir=tmp_path))
    failures = asyncio.run(rec.run())
    assert set(failures) == {"pocketoption"}
    assert isinstance(failures["pocketoption"], ProtocolNotConfigured)
    assert rec.rows_written == 0


def test_recorder_reports_no_failures_on_a_clean_run(tmp_path):
    import asyncio

    from pobot.capture.recorder import Recorder, RecorderConfig

    class Finite(RandomWalkFeed):
        """Ends after a fixed number of ticks instead of streaming forever."""

        async def stream(self):
            for tick in self.generate(50):
                self.stats.record(tick)
                yield tick

    rec = Recorder([Finite(["X"], seed=1)], RecorderConfig(out_dir=tmp_path))
    assert asyncio.run(rec.run()) == {}
    assert rec.rows_written == 50
