"""Pocket Option quote feed.

    THE PROTOCOL SPEC BELOW IS A TEMPLATE, NOT A VERIFIED WIRE FORMAT.

The endpoint, envelope, and field names are undocumented, vary by region and
account, and change without notice. They are left blank on purpose: an empty
`url` makes `stream()` raise `ProtocolNotConfigured` immediately, which is a
loud failure you fix in minutes. A plausible-looking guess would instead connect,
decode nothing, report a healthy feed, and hand you weeks of empty capture files.

Fill it in by observation — see docs/PROTOCOL.md for the ten-minute procedure.

Before you run this against a live account, two things worth knowing:

* Automated trading and programmatic access are restricted by Pocket Option's
  terms, which reserve the right to void trades and close accounts. Capturing a
  public quote stream is the low-risk end of that spectrum; placing orders
  programmatically is the high-risk end. This package deliberately implements
  only capture and offline research — there is no order-placement code here, and
  adding it is a decision to make with the terms in front of you.

* Verify withdrawals work, with a small amount, before scaling anything. An
  unwithdrawable balance is not profit.
"""

from __future__ import annotations

from .wsfeed import ProtocolSpec, SpecDrivenWebSocketFeed

# --------------------------------------------------------------------------
# VERIFY EVERY FIELD BELOW AGAINST LIVE TRAFFIC BEFORE CAPTURING.
# Values are illustrative of the *shape* a Socket.IO-style stream tends to take.
# They are not claimed to be correct for this or any broker.
# --------------------------------------------------------------------------
POCKET_OPTION_SPEC = ProtocolSpec(
    url="",  # <- REQUIRED. e.g. "wss://.../socket.io/?EIO=4&transport=websocket"
    headers={
        # Broker sockets commonly reject connections whose Origin does not match
        # the site, and some require the session cookie from a logged-in browser.
        # "Origin": "https://pocketoption.com",
        # "Cookie": "...",
    },
    handshake=[
        # Socket.IO v4 clients answer the server's "0" open packet with "40",
        # then emit subscribe events as '42["event",payload]'.
        # '40',
        # '42["subscribeSymbol",{symbols}]',
    ],
    keepalive_msg=None,  # Socket.IO v4 answers server "2" pings with "3"
    keepalive_s=20.0,
    strip_prefix="0123456789",  # drop a leading Socket.IO packet-type number
    quotes_path="1",  # '42["stream", <payload>]' -> element 1 is the payload
    symbol_key="asset",
    price_key="price",
    ts_key="time",
    ts_scale=1000.0,  # seconds -> ms. CHECK THIS: a wrong scale shifts every label.
)


class PocketOptionFeed(SpecDrivenWebSocketFeed):
    """Broker-side quote stream — the prices your contracts actually settle on.

    This is the feed that matters for settlement. The reference feed exists only
    to measure how this one behaves relative to a real market.
    """

    def __init__(self, symbols: list[str], spec: ProtocolSpec | None = None) -> None:
        super().__init__(symbols, spec or POCKET_OPTION_SPEC, name="pocketoption")
