"""Spec-driven WebSocket feed.

Both live sources in this project — the broker stream and the independent
reference stream — are JSON-over-WebSocket with different envelopes. Rather than
hardcode either protocol, the wire format is described by a `ProtocolSpec` that
you fill in from observed traffic. Nothing here guesses at a message format.

Why it is built this way: broker WebSocket protocols are undocumented, change
without notice, and differ per account tier and region. Code that hardcodes a
frame layout appears to work, silently decodes nothing after the next
deployment, and leaves you with weeks of empty Parquet files. A spec you can
re-derive in ten minutes from browser DevTools is more robust than a decoder
someone wrote once from memory.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Optional

from .base import FeedAdapter, Tick, now_ms

log = logging.getLogger(__name__)


class ProtocolNotConfigured(RuntimeError):
    """Raised when a feed is started without a verified wire format.

    Deliberately fatal. The alternative — falling back to a guessed layout —
    produces a feed that connects, reports healthy, and records nothing usable.
    """


@dataclass
class ProtocolSpec:
    """Description of a JSON-over-WebSocket price stream.

    Populate from real observed traffic (DevTools -> Network -> WS -> Messages).
    Every field below is a question to answer by looking, not by guessing.
    """

    #: WebSocket endpoint, e.g. "wss://host/socket.io/?EIO=4&transport=websocket".
    url: str = ""

    #: Extra headers. Broker sockets commonly reject connections lacking an
    #: Origin that matches the site.
    headers: dict[str, str] = field(default_factory=dict)

    #: Messages sent immediately after connect, in order. Use "{symbols}" as a
    #: placeholder for the JSON array of subscribed symbols. Some servers expect
    #: an auth frame first; if so it belongs here, ahead of the subscribe frame.
    handshake: list[str] = field(default_factory=list)

    #: Sent every `keepalive_s` seconds. Socket.IO-style servers drop clients
    #: that fail to answer pings.
    keepalive_msg: Optional[str] = None
    keepalive_s: float = 20.0

    #: Some transports prefix frames with a numeric packet type (Socket.IO uses
    #: "42[...]"). Characters to strip before JSON parsing.
    strip_prefix: str = ""

    #: Dotted path to the list of quote objects inside a decoded message, or ""
    #: if the message *is* the quote object. List indices are numeric segments,
    #: e.g. "1.data".
    quotes_path: str = ""

    #: Keys within a single quote object.
    symbol_key: str = "asset"
    price_key: str = "price"
    #: Source-side timestamp. Leave empty if the feed publishes none, in which
    #: case ts_event is set to arrival time and observed latency is always 0 —
    #: which disables lag analysis for this source, so prefer a feed that has one.
    ts_key: str = "time"
    #: Multiplier converting the source timestamp to milliseconds. Seconds -> 1000,
    #: milliseconds -> 1, microseconds -> 0.001. Getting this wrong shifts every
    #: label in the backtest, so verify against a known wall clock.
    ts_scale: float = 1000.0

    bid_key: Optional[str] = None
    ask_key: Optional[str] = None

    def configured(self) -> bool:
        return bool(self.url)


def _dig(obj: Any, path: str) -> Any:
    """Walk a dotted path through nested dicts/lists. Returns None if absent."""
    if not path:
        return obj
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if part.isdigit() and isinstance(cur, (list, tuple)):
            idx = int(part)
            if idx >= len(cur):
                return None
            cur = cur[idx]
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


class SpecDrivenWebSocketFeed(FeedAdapter):
    """Connects, subscribes, decodes per spec, reconnects with backoff.

    Pure transport: it does not filter, resample, or repair. Malformed frames
    increment an error counter and are skipped, because a source that starts
    emitting garbage is a fact worth seeing in the stats rather than hiding.
    """

    name = "ws"

    def __init__(
        self,
        symbols: list[str],
        spec: ProtocolSpec,
        *,
        name: Optional[str] = None,
        max_backoff_s: float = 60.0,
    ) -> None:
        super().__init__(symbols)
        self.spec = spec
        if name:
            self.name = name
        self.max_backoff_s = max_backoff_s

    def decode(self, raw: str | bytes) -> list[Tick]:
        """Turn one wire frame into zero or more ticks.

        Zero is the common case — heartbeats, acks, and control frames all
        decode to nothing and are not errors.
        """
        recv = now_ms()
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                self.stats.errors += 1
                return []

        text = raw
        if self.spec.strip_prefix:
            text = text.lstrip(self.spec.strip_prefix)
        text = text.strip()
        if not text or text[0] not in "[{":
            return []

        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            return []

        quotes = _dig(msg, self.spec.quotes_path)
        if quotes is None:
            return []
        if isinstance(quotes, dict):
            quotes = [quotes]
        if not isinstance(quotes, list):
            return []

        out: list[Tick] = []
        for q in quotes:
            if not isinstance(q, dict):
                continue
            symbol = q.get(self.spec.symbol_key)
            price = q.get(self.spec.price_key)
            if symbol is None or price is None:
                continue
            try:
                price = float(price)
            except (TypeError, ValueError):
                self.stats.errors += 1
                continue

            if self.spec.ts_key:
                rawts = q.get(self.spec.ts_key)
                try:
                    ts_event = int(float(rawts) * self.spec.ts_scale)
                except (TypeError, ValueError):
                    ts_event = recv
            else:
                ts_event = recv

            def _opt(key: Optional[str]) -> Optional[float]:
                if not key:
                    return None
                v = q.get(key)
                try:
                    return float(v) if v is not None else None
                except (TypeError, ValueError):
                    return None

            out.append(
                Tick(
                    symbol=str(symbol),
                    price=price,
                    ts_event=ts_event,
                    ts_recv=recv,
                    source=self.name,
                    bid=_opt(self.spec.bid_key),
                    ask=_opt(self.spec.ask_key),
                )
            )
        return out

    async def _keepalive(self, ws) -> None:
        assert self.spec.keepalive_msg is not None
        while True:
            await asyncio.sleep(self.spec.keepalive_s)
            await ws.send(self.spec.keepalive_msg)

    async def stream(self) -> AsyncIterator[Tick]:
        if not self.spec.configured():
            raise ProtocolNotConfigured(
                f"{self.name}: ProtocolSpec.url is empty. Capture the live wire "
                "format first (see docs/PROTOCOL.md) — this feed will not guess."
            )

        import websockets  # imported lazily so offline tests need no network stack

        backoff = 1.0
        symbols_json = json.dumps(self.symbols)

        while True:
            ka: Optional[asyncio.Task] = None
            try:
                async with websockets.connect(
                    self.spec.url,
                    additional_headers=self.spec.headers or None,
                    ping_interval=None,  # servers here expect app-level keepalive
                    max_queue=4096,
                ) as ws:
                    for frame in self.spec.handshake:
                        await ws.send(frame.replace("{symbols}", symbols_json))
                    if self.spec.keepalive_msg:
                        ka = asyncio.create_task(self._keepalive(ws))

                    backoff = 1.0  # reset only after a successful connect
                    async for raw in ws:
                        for tick in self.decode(raw):
                            self.stats.record(tick)
                            yield tick
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - transport errors are expected
                self.stats.errors += 1
                self.stats.reconnects += 1
                log.warning("%s: connection lost (%s); retry in %.1fs", self.name, exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff_s)
            finally:
                if ka is not None:
                    ka.cancel()
