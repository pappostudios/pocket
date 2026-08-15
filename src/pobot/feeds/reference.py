"""Independent reference feed.

The reason this exists, and why capture is worthless without it:

The broker's own stream cannot be used to audit the broker's own stream. If its
quotes lag a real market, drift, or smooth out spikes, that is invisible when
the broker feed is your only record — every comparison is the series against
itself. A second, unrelated source carrying the same instrument makes the
question answerable.

Two analyses depend on having recorded both, and *neither* can be run
retroactively on broker-only data:

1. Lag estimation. Cross-correlate the two price series at a range of offsets.
   A consistent nonzero peak means the broker feed is a delayed copy — the one
   structurally sound edge in this market, and also the one brokers explicitly
   act against, so read the terms before acting on it.

2. Synthetic-series detection. On "OTC" assets, the broker publishes prices when
   the real market is closed. If a reference series exists and the two decouple,
   the broker's is generated rather than observed — which is what makes the
   mean-reversion hypothesis in feeds/synthetic.py worth testing.

Any source works as long as it is genuinely independent of the broker: a retail
FX broker's public stream, an exchange feed, a market-data vendor. Point the spec
at one you can access.
"""

from __future__ import annotations

from .wsfeed import ProtocolSpec, SpecDrivenWebSocketFeed

# Fill in for whichever independent source you use. Same rule as the broker
# spec: an empty url fails loudly rather than recording nothing.
REFERENCE_SPEC = ProtocolSpec(
    url="",
    quotes_path="",
    symbol_key="s",
    price_key="p",
    ts_key="t",
    ts_scale=1.0,  # many exchange feeds publish milliseconds already
)


class ReferenceFeed(SpecDrivenWebSocketFeed):
    def __init__(
        self,
        symbols: list[str],
        spec: ProtocolSpec | None = None,
        *,
        name: str = "reference",
    ) -> None:
        super().__init__(symbols, spec or REFERENCE_SPEC, name=name)
