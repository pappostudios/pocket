# Deriving the wire format

`feeds/pocketoption.py` ships with an empty `ProtocolSpec`. That is deliberate:
broker WebSocket protocols are undocumented, differ by region and account tier,
and change without notice. A hardcoded guess connects, decodes nothing, reports
a healthy feed, and hands you weeks of empty Parquet files. An empty spec raises
`ProtocolNotConfigured` on the first call instead.

Deriving it takes about ten minutes.

## Procedure

1. Log in to the platform in a browser with an ordinary trading chart open.
2. Open DevTools → Network → filter **WS** → click the socket → **Messages**.
3. Watch the frames for ten seconds. Quote frames are the ones arriving several
   times per second with a number that tracks the chart.
4. Fill in `POCKET_OPTION_SPEC` from what you see:

| Field | What to look for |
|---|---|
| `url` | The socket's request URL, verbatim including query string |
| `headers` | Add `Origin` if the connection is rejected without it |
| `handshake` | Frames the *client* sends right after connect, in order |
| `keepalive_msg` | The client's reply to the server's periodic ping |
| `strip_prefix` | Leading packet-type digits, e.g. the `42` in `42[...]` |
| `quotes_path` | Dotted path from the decoded JSON to the quote list |
| `symbol_key` / `price_key` / `ts_key` | Keys inside one quote object |
| `ts_scale` | `1000` if the timestamp is in seconds, `1` if milliseconds |

5. Verify before capturing:

```python
from pobot.feeds.pocketoption import POCKET_OPTION_SPEC, PocketOptionFeed
feed = PocketOptionFeed(["EURUSD"], POCKET_OPTION_SPEC)
print(feed.decode('<paste one real frame here>'))
```

A correctly specified feed returns one `Tick` per quote. An empty list means the
spec does not match the frame.

## Check `ts_scale` against a wall clock

The single most damaging field to get wrong, and the failure is silent.

```python
from pobot.feeds.base import now_ms
tick = feed.decode(frame)[0]
print("skew:", tick.ts_recv - tick.ts_event, "ms")  # expect roughly 0-500ms
```

Tens of thousands of milliseconds, or a negative number, means the scale is
wrong. Every label in every backtest is then shifted by a constant, and the
results will look plausible while being nonsense.

## Symbol naming

Broker symbols rarely match reference-feed symbols (`EURUSD_otc` vs `EUR/USD`
vs `EURUSD`). Record whatever each source emits — do not normalise during
capture — and map them at analysis time, where the mapping is visible and
revisable.

## When it breaks

It will, because the protocol is not a contract. `pobot summary` shows per-source
tick counts; a source that stops appearing has had its format changed. Re-derive
the spec with the procedure above. This is why the spec is data rather than code:
re-deriving takes ten minutes, rewriting a decoder does not.

## Before pointing this at a live account

Automated trading and programmatic access are restricted by Pocket Option's
terms, which reserve the right to void trades and close accounts. Capturing a
public quote stream sits at the low-risk end of that spectrum; placing orders
programmatically sits at the high-risk end. This package implements capture and
offline research only — there is no order-placement code in it, and adding some
is a decision to make with the terms in front of you.

Separately: verify a withdrawal works, with a small amount, before scaling
anything. The binding constraint on this kind of account is usually getting
money out, not the strategy.
