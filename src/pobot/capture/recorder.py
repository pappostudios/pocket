"""Multi-feed recorder.

Runs every feed concurrently, each with its own writer, and supervises them. One
feed dying must not take down the others: the broker feed and the reference feed
fail for unrelated reasons, and a night of single-feed capture still beats none.

Heartbeat logging is deliberately verbose. Weeks of unattended capture with no
health output is how you discover a dead feed in week 4 instead of on day 3.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path

from ..feeds.base import FeedAdapter, now_ms
from .writer import TickWriter

log = logging.getLogger(__name__)


@dataclass
class RecorderConfig:
    out_dir: Path
    buffer_rows: int = 5000
    flush_interval_s: float = 30.0
    heartbeat_interval_s: float = 60.0
    #: Warn when a feed has produced nothing for this long. Not fatal — a quiet
    #: market and a dead socket look identical from here, and only you know which
    #: is plausible at 03:00 UTC on a Sunday.
    stale_warn_s: float = 120.0


class Recorder:
    def __init__(self, feeds: list[FeedAdapter], config: RecorderConfig) -> None:
        self.feeds = feeds
        self.config = config
        self.writers: dict[str, TickWriter] = {}
        #: Feeds that terminated permanently, by name. Read after `run()` —
        #: a recorder that captured nothing must not look like a success.
        self.failures: dict[str, BaseException] = {}

    async def _drain(self, feed: FeedAdapter) -> None:
        from ..feeds.wsfeed import ProtocolNotConfigured

        writer = TickWriter(
            self.config.out_dir, feed.name, buffer_rows=self.config.buffer_rows
        )
        self.writers[feed.name] = writer
        try:
            async for tick in feed.stream():
                writer.append(tick)
        except asyncio.CancelledError:
            raise
        except ProtocolNotConfigured as exc:
            # A configuration error, not a crash. The message is the whole
            # story, so a traceback would only bury it.
            log.error("feed %s: %s", feed.name, exc)
            self.failures[feed.name] = exc
            raise
        except Exception as exc:
            log.exception("feed %s terminated permanently", feed.name)
            self.failures[feed.name] = exc
            raise
        finally:
            writer.close()
            log.info("feed %s: wrote %d rows", feed.name, writer.rows_written)

    async def _periodic_flush(self) -> None:
        """Bound data loss on an unclean exit to one flush interval."""
        while True:
            await asyncio.sleep(self.config.flush_interval_s)
            for w in self.writers.values():
                with contextlib.suppress(Exception):
                    w.flush()

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(self.config.heartbeat_interval_s)
            ref = now_ms()
            for feed in self.feeds:
                s = feed.stats
                stale = s.staleness_ms(ref)
                stale_txt = "never" if stale is None else f"{stale / 1000:.0f}s"
                line = (
                    f"{feed.name}: ticks={s.ticks} symbols={len(s.symbols_seen)} "
                    f"reconnects={s.reconnects} errors={s.errors} last={stale_txt}"
                )
                if stale is None or stale > self.config.stale_warn_s * 1000:
                    log.warning("STALE %s", line)
                else:
                    log.info(line)

    @property
    def rows_written(self) -> int:
        return sum(w.rows_written for w in self.writers.values())

    async def run(self) -> dict[str, BaseException]:
        """Record until every feed ends or the task is cancelled.

        Returns the map of permanently-failed feeds. Callers must check it:
        gathering with `return_exceptions=True` means a total failure would
        otherwise return normally and look like a successful capture.
        """
        self.config.out_dir.mkdir(parents=True, exist_ok=True)
        tasks = [asyncio.create_task(self._drain(f), name=f"feed:{f.name}") for f in self.feeds]
        aux = [
            asyncio.create_task(self._periodic_flush(), name="flush"),
            asyncio.create_task(self._heartbeat(), name="heartbeat"),
        ]
        try:
            # Wait for all feeds; a single failure is logged by _drain and the
            # remaining feeds keep recording.
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            for t in aux:
                t.cancel()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*aux, *tasks, return_exceptions=True)
            for w in self.writers.values():
                with contextlib.suppress(Exception):
                    w.close()
        return self.failures
