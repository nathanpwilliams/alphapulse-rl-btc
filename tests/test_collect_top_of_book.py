"""
tests/test_collect_top_of_book.py - Unit tests for the Phase B TOB collector.

Synthetic messages modeled on the venues' real (probed) schemas, no network. The
async I/O is verified by the live smoke run, not here; these tests pin the pure
parsers, symbol mapping, and the append-only daily-CSV sink.

Run:
    pytest tests/test_collect_top_of_book.py -v
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data import collect_top_of_book as c  # noqa: E402


# ============================================================
# Symbol mapping
# ============================================================

class TestSymbolMapping:

    def test_base_to_venue(self):
        assert c.base_to_kraken("BTC") == "PF_XBTUSD"
        assert c.base_to_kraken("ARB") == "PF_ARBUSD"
        assert c.base_to_okx("BTC") == "BTC-USDT-SWAP"

    def test_venue_to_base_roundtrip(self):
        assert c.kraken_to_base("PF_XBTUSD") == "BTC"
        assert c.kraken_to_base("PF_ARBUSD") == "ARB"
        assert c.okx_to_base("ARB-USDT-SWAP") == "ARB"


# ============================================================
# Kraken ticker parsing
# ============================================================

class TestKrakenParse:

    def _ticker(self, **kw):
        base = {"feed": "ticker", "product_id": "PF_ARBUSD", "bid": 0.22109,
                "ask": 0.22148, "bid_size": 6000.0, "ask_size": 6779.0, "time": 1790279657334}
        base.update(kw)
        return base

    def test_parses_ticker(self):
        row = c.parse_kraken_ticker(self._ticker(), ts_recv_ms=1790279657400)
        assert row["venue"] == "kraken"
        assert row["asset"] == "ARB"
        assert row["bid"] == pytest.approx(0.22109)
        assert row["ask_size"] == pytest.approx(6779.0)
        assert row["ts_exchange"] == 1790279657334
        assert row["ts_recv"] == 1790279657400

    def test_non_ticker_returns_none(self):
        assert c.parse_kraken_ticker({"feed": "heartbeat"}, 1) is None
        assert c.parse_kraken_ticker({"event": "subscribed", "feed": "ticker"}, 1) is None

    def test_missing_bid_returns_none(self):
        assert c.parse_kraken_ticker(self._ticker(bid=None), 1) is None


# ============================================================
# OKX bbo-tbt parsing
# ============================================================

class TestOkxParse:

    def _bbo(self):
        return {"arg": {"channel": "bbo-tbt", "instId": "ARB-USDT-SWAP"},
                "data": [{"asks": [["0.22106", "562.1", "0", "14"]],
                          "bids": [["0.22105", "897", "0", "5"]],
                          "seqId": 1, "ts": "1790279658607"}]}

    def test_parses_bbo(self):
        rows = c.parse_okx_bbo(self._bbo(), ts_recv_ms=1790279658700)
        assert len(rows) == 1
        r = rows[0]
        assert r["venue"] == "okx" and r["asset"] == "ARB"
        assert r["bid"] == pytest.approx(0.22105)
        assert r["ask"] == pytest.approx(0.22106)
        assert r["bid_size"] == pytest.approx(897.0)
        assert r["ts_exchange"] == 1790279658607

    def test_ack_frame_returns_empty(self):
        assert c.parse_okx_bbo({"event": "subscribe", "arg": {"channel": "bbo-tbt"}}, 1) == []
        assert c.parse_okx_bbo({"arg": {"channel": "bbo-tbt", "instId": "X-USDT-SWAP"}, "data": []}, 1) == []

    def test_tickers_fallback_schema(self):
        msg = {"arg": {"channel": "tickers", "instId": "ARB-USDT-SWAP"},
               "data": [{"bidPx": "0.221", "bidSz": "10", "askPx": "0.222", "askSz": "12", "ts": "1790279658607"}]}
        rows = c.parse_okx_bbo(msg, 1)
        assert len(rows) == 1 and rows[0]["ask"] == pytest.approx(0.222)


# ============================================================
# Daily CSV sink
# ============================================================

class TestAppendRows:

    def test_writes_header_and_rows(self, tmp_path):
        rows = [
            {"ts_recv": 1704067200000, "ts_exchange": 1, "venue": "kraken", "asset": "BTC",
             "bid": 1.0, "bid_size": 2.0, "ask": 1.1, "ask_size": 3.0},
        ]
        c.append_rows(rows, tmp_path, "tob", c.TOB_COLUMNS)
        files = list(tmp_path.glob("tob_*.csv"))
        assert len(files) == 1
        df = pd.read_csv(files[0])
        assert list(df.columns) == c.TOB_COLUMNS
        assert len(df) == 1 and df["asset"].iloc[0] == "BTC"

    def test_appends_without_duplicate_header(self, tmp_path):
        row = {"ts_recv": 1704067200000, "ts_exchange": 1, "venue": "okx", "asset": "ARB",
               "bid": 1.0, "bid_size": 2.0, "ask": 1.1, "ask_size": 3.0}
        c.append_rows([row], tmp_path, "tob", c.TOB_COLUMNS)
        c.append_rows([row], tmp_path, "tob", c.TOB_COLUMNS)
        df = pd.read_csv(next(tmp_path.glob("tob_*.csv")))
        assert len(df) == 2  # two data rows, one header

    def test_splits_across_utc_days(self, tmp_path):
        # 2024-01-01 23:59 and 2024-01-02 00:01 -> two daily files.
        d1 = int(pd.Timestamp("2024-01-01 23:59:00", tz="UTC").timestamp() * 1000)
        d2 = int(pd.Timestamp("2024-01-02 00:01:00", tz="UTC").timestamp() * 1000)
        rows = [
            {"ts_recv": d1, "ts_exchange": 1, "venue": "kraken", "asset": "BTC",
             "bid": 1, "bid_size": 1, "ask": 1, "ask_size": 1},
            {"ts_recv": d2, "ts_exchange": 1, "venue": "kraken", "asset": "BTC",
             "bid": 1, "bid_size": 1, "ask": 1, "ask_size": 1},
        ]
        c.append_rows(rows, tmp_path, "tob", c.TOB_COLUMNS)
        assert (tmp_path / "tob_20240101.csv").exists()
        assert (tmp_path / "tob_20240102.csv").exists()

    def test_empty_rows_noop(self, tmp_path):
        c.append_rows([], tmp_path, "tob", c.TOB_COLUMNS)
        assert list(tmp_path.glob("*.csv")) == []


# ============================================================
# Graceful shutdown / final flush
# ============================================================
#
# `systemctl stop` sends SIGTERM, whose DEFAULT action kills the interpreter
# outright -- no `finally`, no final flush. The Phase B run lost ~200 buffered
# rows that way on 2026-10-13. Every test here sets `flush_interval_sec` far
# longer than the test runs, so the periodic writer can NEVER fire: if rows
# appear on disk, only the final flush in `run_collector`'s `finally` put them
# there.

def _quote_row(ts_ms: int = 1704067200000, asset: str = "BTC") -> dict:
    return {"ts_recv": ts_ms, "ts_exchange": ts_ms - 20, "venue": "kraken",
            "asset": asset, "bid": 1.0, "bid_size": 2.0, "ask": 1.1, "ask_size": 3.0}


def _buffering_feed(rows, event_detail="stub"):
    """A feed stub that buffers `rows` once, then idles forever."""
    async def feed(config, buffer, events):
        buffer.extend(rows)
        events.append({"ts_recv": rows[0]["ts_recv"], "venue": "kraken", "asset": "*",
                       "event": "connected", "detail": event_detail})
        while True:
            await asyncio.sleep(3600)
    return feed


async def _idle_feed(config, buffer, events):
    while True:
        await asyncio.sleep(3600)


def _run_with_signal(config, sig, delay=0.05):
    """Run the collector and deliver `sig` to our own process from the loop."""
    async def driver():
        loop = asyncio.get_running_loop()
        loop.call_later(delay, os.kill, os.getpid(), sig)
        await c.run_collector(config)
    asyncio.run(driver())


class TestGracefulShutdown:

    @pytest.fixture(autouse=True)
    def _stub_feeds(self, monkeypatch):
        self.rows = [_quote_row(), _quote_row(asset="ARB")]
        monkeypatch.setattr(c, "_kraken_feed", _buffering_feed(self.rows))
        monkeypatch.setattr(c, "_okx_feed", _idle_feed)

    def _config(self, tmp_path, **kw):
        kw.setdefault("duration_sec", 0.0)
        return c.CollectorConfig(out_dir=tmp_path, flush_interval_sec=3600.0, **kw)

    @pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
    def test_signal_runs_the_final_flush(self, tmp_path, sig):
        """The regression test: SIGTERM/SIGINT must flush, not kill the process."""
        _run_with_signal(self._config(tmp_path), sig)

        df = pd.read_csv(next(tmp_path.glob("tob_2*.csv")))
        assert len(df) == 2
        assert set(df["asset"]) == {"BTC", "ARB"}

    def test_signal_also_flushes_the_events_log(self, tmp_path):
        _run_with_signal(self._config(tmp_path), signal.SIGTERM)

        ev = pd.read_csv(next(tmp_path.glob("tob_events_*.csv")))
        assert list(ev.columns) == c.EVENT_COLUMNS
        assert ev["event"].iloc[0] == "connected"

    def test_signal_returns_rather_than_raising(self, tmp_path):
        """A clean stop is not an error: `systemctl stop` should exit 0."""
        _run_with_signal(self._config(tmp_path), signal.SIGTERM)  # must not raise

    def test_duration_elapsing_also_flushes(self, tmp_path):
        """The pre-existing smoke-test path must keep working."""
        asyncio.run(c.run_collector(self._config(tmp_path, duration_sec=0.1)))

        df = pd.read_csv(next(tmp_path.glob("tob_2*.csv")))
        assert len(df) == 2

    def test_handlers_are_restored_after_shutdown(self, tmp_path):
        """Leaving a handler installed would silently break the next run in-process."""
        before = {s: signal.getsignal(s) for s in c.SHUTDOWN_SIGNALS}

        _run_with_signal(self._config(tmp_path), signal.SIGTERM)

        assert {s: signal.getsignal(s) for s in c.SHUTDOWN_SIGNALS} == before

    def test_feed_crash_propagates_and_still_flushes(self, tmp_path, monkeypatch):
        """A dying feed must surface (as `gather` used to) without losing the buffer."""
        rows = self.rows

        async def dying_feed(config, buffer, events):
            buffer.extend(rows)
            await asyncio.sleep(0.01)
            raise RuntimeError("feed died")

        monkeypatch.setattr(c, "_kraken_feed", dying_feed)

        with pytest.raises(RuntimeError, match="feed died"):
            asyncio.run(c.run_collector(self._config(tmp_path)))

        df = pd.read_csv(next(tmp_path.glob("tob_2*.csv")))
        assert len(df) == 2

    def test_periodic_writer_is_not_what_wrote_the_rows(self, tmp_path):
        """Guard the guard: with a 1 h flush interval nothing writes on its own."""
        async def driver():
            config = self._config(tmp_path, duration_sec=0.05)
            task = asyncio.create_task(c.run_collector(config))
            await asyncio.sleep(0.02)
            assert list(tmp_path.glob("tob_2*.csv")) == []   # nothing yet
            await task

        asyncio.run(driver())
        assert len(list(tmp_path.glob("tob_2*.csv"))) == 1   # only the final flush


class TestShutdownHandlerInstall:

    def test_install_and_remove_round_trip(self):
        async def check():
            loop = asyncio.get_running_loop()
            stop = asyncio.Event()
            installed = c._install_shutdown_handlers(loop, stop)
            assert set(installed) == set(c.SHUTDOWN_SIGNALS)
            os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.sleep(0.05)
            assert stop.is_set()
            c._remove_shutdown_handlers(loop, installed)

        asyncio.run(check())

    def test_install_tolerates_an_unsupported_platform(self, monkeypatch):
        """Windows / non-main-thread: warn and fall back, never crash."""
        async def check():
            loop = asyncio.get_running_loop()
            monkeypatch.setattr(
                loop, "add_signal_handler",
                lambda *a, **k: (_ for _ in ()).throw(NotImplementedError()))
            assert c._install_shutdown_handlers(loop, asyncio.Event()) == []

        asyncio.run(check())
