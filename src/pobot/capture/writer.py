"""Buffered Parquet writer.

Capture runs for weeks. The failure mode that matters is not throughput, it is
losing a partially-written file to a crash on day 19 and finding out on day 21.
So: fixed-size row-group flushes, one file per (source, UTC hour), fsync on
close, and a `.tmp` -> final rename so a killed process can never leave a
half-written file that looks complete to the reader.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from ..feeds.base import Tick

log = logging.getLogger(__name__)

SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("ts_event", pa.int64()),
        pa.field("ts_recv", pa.int64()),
        pa.field("source", pa.string()),
        pa.field("bid", pa.float64()),
        pa.field("ask", pa.float64()),
    ]
)


def _hour_key(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y%m%dT%H")


class TickWriter:
    """Append ticks, flushing to hour-partitioned Parquet.

    Not thread-safe; the recorder gives each feed its own writer, which also
    keeps a slow or dead feed from blocking a healthy one.
    """

    def __init__(self, root: Path | str, source: str, *, buffer_rows: int = 5000) -> None:
        self.root = Path(root)
        self.source = source
        self.buffer_rows = buffer_rows
        self._buf: list[Tick] = []
        self._writer: Optional[pq.ParquetWriter] = None
        self._hour: Optional[str] = None
        self._tmp_path: Optional[Path] = None
        self._final_path: Optional[Path] = None
        self.rows_written = 0

    def _paths_for(self, hour: str) -> tuple[Path, Path]:
        d = self.root / f"source={self.source}" / f"hour={hour}"
        d.mkdir(parents=True, exist_ok=True)
        return d / "ticks.parquet.tmp", d / "ticks.parquet"

    def _roll(self, hour: str) -> None:
        self._close_current()
        self._hour = hour
        self._tmp_path, self._final_path = self._paths_for(hour)
        # An existing final file means a previous run already covered this hour.
        # Suffix rather than overwrite: captured data is never silently destroyed.
        if self._final_path.exists():
            n = 1
            while True:
                cand = self._final_path.with_name(f"ticks.{n}.parquet")
                if not cand.exists():
                    self._final_path = cand
                    self._tmp_path = cand.with_suffix(".parquet.tmp")
                    break
                n += 1
        self._writer = pq.ParquetWriter(self._tmp_path, SCHEMA, compression="zstd")

    def _close_current(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None
        if self._tmp_path and self._final_path and self._tmp_path.exists():
            os.replace(self._tmp_path, self._final_path)
            log.info("sealed %s", self._final_path)

    def append(self, tick: Tick) -> None:
        self._buf.append(tick)
        if len(self._buf) >= self.buffer_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        # Partition by arrival time: it is monotonic and locally controlled,
        # whereas a source-supplied ts_event can jump backwards on reconnect and
        # would scatter one hour's data across several files.
        hour = _hour_key(self._buf[0].ts_recv)
        if hour != self._hour:
            self._roll(hour)

        # A buffer spanning an hour boundary is written whole to the file its
        # first row selected. Readers filter on ts_recv, never on the partition
        # key, so a few rows landing in a neighbouring file changes no result.
        table = pa.Table.from_pydict(
            {
                "symbol": [t.symbol for t in self._buf],
                "price": [t.price for t in self._buf],
                "ts_event": [t.ts_event for t in self._buf],
                "ts_recv": [t.ts_recv for t in self._buf],
                "source": [t.source for t in self._buf],
                "bid": [t.bid for t in self._buf],
                "ask": [t.ask for t in self._buf],
            },
            schema=SCHEMA,
        )
        assert self._writer is not None
        self._writer.write_table(table)
        self.rows_written += len(self._buf)
        self._buf.clear()

    def close(self) -> None:
        self.flush()
        self._close_current()

    def __enter__(self) -> "TickWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
