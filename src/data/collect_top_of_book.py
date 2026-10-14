"""
src/data/collect_top_of_book.py - Live cross-venue top-of-book collector (Phase B).

Phase A's candle screen nominated ARB/NEAR/UNI but could not prove EXECUTABILITY
(half-life below the 5m sampling interval, stale last-trade prints, unseen
spreads). Phase B records live best bid/ask from BOTH venues so the
executable-gap analyzer can decide whether real, tradeable dislocations survive.

This module ONLY collects public market data. No auth, no orders, no trading.

Feeds (public, confirmed live):
    Kraken Futures  wss://futures.kraken.com/ws/v1   feed="ticker"
        {"feed":"ticker","product_id":"PF_ARBUSD","bid":..,"ask":..,
         "bid_size":..,"ask_size":..,"time":<ms>, ...}   (bid/ask numeric; time ms)
    OKX             wss://ws.okx.com:8443/ws/v5/public channel="bbo-tbt"
        {"arg":{"channel":"bbo-tbt","instId":"ARB-USDT-SWAP"},
         "data":[{"asks":[[px,sz,"0",nOrders]],"bids":[[..]],"ts":"<ms>"}]}  (strings)

Design:
    * Each update -> a row: ts_recv (collector wall clock, UTC ms - the ONE clock
      used for cross-venue alignment), ts_exchange, venue, asset, bid, bid_size,
      ask, ask_size. Appended to daily CSV under out_dir (CSV, not Parquet, so a
      long-lived process appends line-by-line and survives a crash mid-file).
    * A SEPARATE daily events CSV logs connects/disconnects/timeouts: a feed
      outage is itself data - the analyzer must not read a dead feed as a quiet
      market.
    * Auto-reconnect with exponential backoff; OKX app-level ping; a recv timeout
      is treated as a dead connection and triggers reconnect.
    * Graceful shutdown: SIGTERM/SIGINT set a stop event so the final flush runs.
      Python's DEFAULT SIGTERM action kills the interpreter outright, skipping
      every `finally` -- and `systemctl stop` sends SIGTERM.

Run (long-lived; see docs/tob_collector_runbook.md for droplet deployment):
    python -m src.data.collect_top_of_book --symbols ARB,NEAR,UNI,BTC --out data/tob/
    python -m src.data.collect_top_of_book --symbols ARB,BTC --duration 120   # smoke
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

KRAKEN_WS_URL = "wss://futures.kraken.com/ws/v1"
OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

# Published taker fees (perpetuals, base tier), for the analyzer's default cost.
# Kraken Futures taker 0.050% -> 5 bps (kraken.com/features/fee-schedule#futures).
# OKX perpetual taker 0.050% -> 5 bps, regular Lv1 (okx.com/fees). Verify current
# schedule before trusting downstream numbers.
KRAKEN_TAKER_BPS = 5.0
OKX_TAKER_BPS = 5.0

TOB_COLUMNS = ["ts_recv", "ts_exchange", "venue", "asset", "bid", "bid_size", "ask", "ask_size"]
EVENT_COLUMNS = ["ts_recv", "venue", "asset", "event", "detail"]

DEFAULT_SYMBOLS = ["ARB", "NEAR", "UNI", "BTC"]

# ------------------------------------------------------------
# Size units: the two venues do NOT agree
# ------------------------------------------------------------
# The collector records each venue's `bid_size`/`ask_size` VERBATIM, and they are
# in different units:
#   * Kraken Futures PF_ perps quote size in the BASE currency (e.g. BTC).
#   * OKX SWAP quotes size in CONTRACTS. Converting to base units needs ctVal
#     (base units per contract), from GET /api/v5/public/instruments?instType=SWAP,
#     read 2026-10-14 (ctMult is 1 on all four):
#         BTC-USDT-SWAP  ctVal 0.01 BTC    NEAR-USDT-SWAP  ctVal 10 NEAR
#         UNI-USDT-SWAP  ctVal 1    UNI    ARB-USDT-SWAP   ctVal 10 ARB
#
# ANY cross-venue size comparison must convert FIRST. Comparing them raw - e.g.
# min(kraken_size, okx_size) - silently mixes units and was a real bug in the
# Phase B capacity figures (OKX BTC notional came out 100x too large).
OKX_CONTRACT_SIZE: dict[str, float] = {"BTC": 0.01, "NEAR": 10.0, "UNI": 1.0, "ARB": 10.0}


def size_multiplier(venue: str, asset: str) -> float:
    """Base units per quoted size unit for (venue, asset).

    Multiply a raw `bid_size`/`ask_size` by this to get base-currency units,
    which is the only form in which the two venues' sizes may be compared.

    Args:
        venue: "kraken" or "okx".
        asset: base asset, e.g. "BTC" (case-insensitive).

    Returns:
        ctVal for an OKX swap; 1.0 for Kraken (already base units). An OKX asset
        missing from `OKX_CONTRACT_SIZE` returns 1.0 and logs a warning - add it
        to the table rather than relying on that fallback.
    """
    if venue.lower() != "okx":
        return 1.0
    mult = OKX_CONTRACT_SIZE.get(asset.upper())
    if mult is None:
        logger.warning("No OKX ctVal for %s; assuming 1.0 base unit/contract, which "
                       "is probably wrong. Add it to OKX_CONTRACT_SIZE.", asset)
        return 1.0
    return mult


# ============================================================
# Symbol mapping
# ============================================================

def base_to_kraken(base: str) -> str:
    """BTC -> PF_XBTUSD; else PF_<BASE>USD (Kraken uses XBT for BTC)."""
    return f"PF_{'XBT' if base.upper() == 'BTC' else base.upper()}USD"


def base_to_okx(base: str) -> str:
    """<BASE>-USDT-SWAP."""
    return f"{base.upper()}-USDT-SWAP"


def kraken_to_base(product_id: str) -> str:
    """PF_<BASE>USD -> BASE (XBT -> BTC)."""
    b = product_id[3:-3].upper() if product_id.upper().startswith("PF_") else product_id.upper()
    return "BTC" if b == "XBT" else b


def okx_to_base(inst_id: str) -> str:
    """<BASE>-USDT-SWAP -> BASE."""
    return inst_id.split("-")[0].upper()


# ============================================================
# Config
# ============================================================

@dataclass
class CollectorConfig:
    """Parameters for the top-of-book collector."""

    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    out_dir: Path = Path("data/tob")
    flush_interval_sec: float = 5.0     # buffer -> disk cadence
    okx_ping_sec: float = 20.0          # OKX drops idle sockets ~30s
    recv_timeout_sec: float = 30.0      # no msg this long => treat as dead
    max_backoff_sec: float = 60.0       # reconnect backoff cap
    duration_sec: float = 0.0           # 0 = run forever; >0 = stop (smoke tests)


# ============================================================
# Pure parsers (no I/O - unit tested)
# ============================================================

def _now_ms() -> int:
    return int(time.time() * 1000)


def parse_kraken_ticker(msg: dict[str, Any], ts_recv_ms: int) -> dict[str, Any] | None:
    """Parse a Kraken Futures `ticker` message to a TOB row, or None if not a quote.

    Non-ticker frames (info/subscribed/heartbeat) and quotes missing bid/ask return
    None so the caller skips them cleanly.
    """
    if msg.get("feed") != "ticker":
        return None
    pid = msg.get("product_id")
    if not pid or msg.get("bid") is None or msg.get("ask") is None:
        return None
    return {
        "ts_recv": ts_recv_ms,
        "ts_exchange": int(msg["time"]) if msg.get("time") is not None else None,
        "venue": "kraken",
        "asset": kraken_to_base(pid),
        "bid": float(msg["bid"]),
        "bid_size": float(msg["bid_size"]) if msg.get("bid_size") is not None else float("nan"),
        "ask": float(msg["ask"]),
        "ask_size": float(msg["ask_size"]) if msg.get("ask_size") is not None else float("nan"),
    }


def parse_okx_bbo(msg: dict[str, Any], ts_recv_ms: int) -> list[dict[str, Any]]:
    """Parse an OKX `bbo-tbt` (or `tickers`) message to TOB rows (possibly several).

    bbo-tbt rows carry `asks`/`bids` as [[px, sz, _, nOrders], ...] strings; the
    `tickers` fallback carries flat bidPx/bidSz/askPx/askSz. Ack/error frames
    (no `data`) yield [].
    """
    arg = msg.get("arg", {})
    channel = arg.get("channel")
    if channel not in ("bbo-tbt", "tickers") or not msg.get("data"):
        return []
    base = okx_to_base(arg.get("instId", ""))
    rows: list[dict[str, Any]] = []
    for d in msg["data"]:
        if "asks" in d and "bids" in d:  # bbo-tbt
            if not d["asks"] or not d["bids"]:
                continue
            ask, bid = d["asks"][0], d["bids"][0]
            row = {
                "bid": float(bid[0]), "bid_size": float(bid[1]),
                "ask": float(ask[0]), "ask_size": float(ask[1]),
            }
        elif "bidPx" in d:  # tickers fallback
            if not d.get("bidPx") or not d.get("askPx"):
                continue
            row = {
                "bid": float(d["bidPx"]), "bid_size": float(d.get("bidSz") or "nan"),
                "ask": float(d["askPx"]), "ask_size": float(d.get("askSz") or "nan"),
            }
        else:
            continue
        row.update({
            "ts_recv": ts_recv_ms,
            "ts_exchange": int(d["ts"]) if d.get("ts") is not None else None,
            "venue": "okx", "asset": base,
        })
        rows.append(row)
    return rows


# ============================================================
# Sinks (append-friendly daily CSV - unit tested)
# ============================================================

def _daily_path(out_dir: Path, prefix: str, ts_recv_ms: int) -> Path:
    day = datetime.fromtimestamp(ts_recv_ms / 1000, tz=timezone.utc).strftime("%Y%m%d")
    return out_dir / f"{prefix}_{day}.csv"


def append_rows(rows: list[dict[str, Any]], out_dir: Path, prefix: str, columns: list[str]) -> None:
    """Append rows to the per-UTC-day CSV for `prefix`, writing a header on create.

    Grouped by each row's ts_recv date so a flush spanning midnight lands correctly.
    """
    if not rows:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_path: dict[Path, list[dict[str, Any]]] = {}
    for r in rows:
        by_path.setdefault(_daily_path(out_dir, prefix, r["ts_recv"]), []).append(r)
    for path, group in by_path.items():
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(group)


# ============================================================
# Async collection (network - smoke-tested, not unit-tested)
# ============================================================

async def _kraken_feed(config: CollectorConfig, buffer: list, events: list) -> None:
    import websockets
    product_ids = [base_to_kraken(s) for s in config.symbols]
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(KRAKEN_WS_URL, ping_interval=None, open_timeout=15) as ws:
                await ws.send(json.dumps({"event": "subscribe", "feed": "ticker",
                                          "product_ids": product_ids}))
                events.append({"ts_recv": _now_ms(), "venue": "kraken", "asset": "*",
                               "event": "connected", "detail": ",".join(product_ids)})
                backoff = 1.0
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=config.recv_timeout_sec)
                    row = parse_kraken_ticker(json.loads(raw), _now_ms())
                    if row is not None:
                        buffer.append(row)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            events.append({"ts_recv": _now_ms(), "venue": "kraken", "asset": "*",
                           "event": "disconnect", "detail": f"{type(e).__name__}: {e}"})
            logger.warning("Kraken feed dropped (%s); reconnecting in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, config.max_backoff_sec)


async def _okx_feed(config: CollectorConfig, buffer: list, events: list) -> None:
    import websockets
    inst_ids = [base_to_okx(s) for s in config.symbols]
    args = [{"channel": "bbo-tbt", "instId": i} for i in inst_ids]
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(OKX_WS_URL, ping_interval=None, open_timeout=15) as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": args}))
                events.append({"ts_recv": _now_ms(), "venue": "okx", "asset": "*",
                               "event": "connected", "detail": ",".join(inst_ids)})
                backoff = 1.0

                async def _pinger():  # OKX closes idle sockets; app-level ping keeps alive
                    while True:
                        await asyncio.sleep(config.okx_ping_sec)
                        await ws.send("ping")

                pinger = asyncio.create_task(_pinger())
                try:
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=config.recv_timeout_sec)
                        if raw == "pong":
                            continue
                        buffer.extend(parse_okx_bbo(json.loads(raw), _now_ms()))
                finally:
                    pinger.cancel()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            events.append({"ts_recv": _now_ms(), "venue": "okx", "asset": "*",
                           "event": "disconnect", "detail": f"{type(e).__name__}: {e}"})
            logger.warning("OKX feed dropped (%s); reconnecting in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, config.max_backoff_sec)


async def _writer(config: CollectorConfig, buffer: list, events: list) -> None:
    """Flush buffered quote rows and event rows to daily CSVs on a cadence."""
    while True:
        await asyncio.sleep(config.flush_interval_sec)
        if buffer:
            rows, buffer[:] = list(buffer), []
            append_rows(rows, config.out_dir, "tob", TOB_COLUMNS)
            logger.info("Flushed %d quote rows (buffer cleared).", len(rows))
        if events:
            evs, events[:] = list(events), []
            append_rows(evs, config.out_dir, "tob_events", EVENT_COLUMNS)


SHUTDOWN_SIGNALS = (signal.SIGTERM, signal.SIGINT)


def _install_shutdown_handlers(
    loop: asyncio.AbstractEventLoop, stop: asyncio.Event
) -> list[signal.Signals]:
    """Make SIGTERM/SIGINT set `stop` instead of killing the process.

    Python's DEFAULT SIGTERM action terminates the interpreter immediately: no
    `finally` blocks, no final flush. `systemctl stop` sends exactly that, so
    without this the buffered rows since the last periodic flush are lost (this
    cost ~200 rows when the Phase B collector was stopped on 2026-10-13).

    Returns the signals actually installed, so they can be restored afterwards.
    `add_signal_handler` is unavailable on some platforms (Windows) and outside
    the main thread; there we leave the default action rather than fail.
    """
    installed: list[signal.Signals] = []
    for sig in SHUTDOWN_SIGNALS:
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError, ValueError, AttributeError):
            logger.warning("Cannot install %s handler; final flush not guaranteed.", sig.name)
            continue
        installed.append(sig)
    return installed


def _remove_shutdown_handlers(
    loop: asyncio.AbstractEventLoop, installed: list[signal.Signals]
) -> None:
    """Restore the default disposition of every handler we installed."""
    for sig in installed:
        try:
            loop.remove_signal_handler(sig)
        except (NotImplementedError, RuntimeError, ValueError):
            pass


async def run_collector(config: CollectorConfig) -> None:
    """Run both feeds + writer until duration elapses (0 = forever) or a signal.

    Stops on whichever comes first: `config.duration_sec` elapsing, SIGTERM or
    SIGINT, or (in forever mode) a feed task raising. In every case the `finally`
    block runs and flushes whatever is still buffered, so at most the in-flight
    rows are lost -- never the file.
    """
    buffer: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed = _install_shutdown_handlers(loop, stop)

    tasks = [
        asyncio.create_task(_kraken_feed(config, buffer, events)),
        asyncio.create_task(_okx_feed(config, buffer, events)),
        asyncio.create_task(_writer(config, buffer, events)),
    ]
    stop_task = asyncio.create_task(stop.wait())
    waiters = [stop_task]
    if config.duration_sec > 0:
        waiters.append(asyncio.create_task(asyncio.sleep(config.duration_sec)))
    else:
        # Forever mode: a dying feed must surface, as `gather` used to make it.
        waiters.extend(tasks)

    try:
        done, _pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        if stop_task in done:
            logger.info("Shutdown signal received; draining buffers.")
        for t in tasks:
            if t in done:
                t.result()  # re-raise a feed's exception, as before
    finally:
        _remove_shutdown_handlers(loop, installed)
        pending = set(tasks) | set(waiters)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        # Final flush of anything still buffered.
        n_quotes, n_events = len(buffer), len(events)
        append_rows(buffer, config.out_dir, "tob", TOB_COLUMNS)
        append_rows(events, config.out_dir, "tob_events", EVENT_COLUMNS)
        logger.info("Collector stopped; final flush complete (%d quote rows, %d events).",
                    n_quotes, n_events)


# ============================================================
# CLI
# ============================================================

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Live cross-venue top-of-book collector (public feeds only).")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                        help="Comma-separated base assets (default ARB,NEAR,UNI,BTC).")
    parser.add_argument("--out", default="data/tob", help="Output directory.")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Seconds to run then stop; 0 = run forever (default).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = CollectorConfig(
        symbols=[s.strip().upper() for s in args.symbols.split(",") if s.strip()],
        out_dir=Path(args.out),
        duration_sec=args.duration,
    )
    logger.info("Collecting TOB for %s -> %s (duration=%s)",
                config.symbols, config.out_dir, args.duration or "forever")
    asyncio.run(run_collector(config))


if __name__ == "__main__":
    main()
