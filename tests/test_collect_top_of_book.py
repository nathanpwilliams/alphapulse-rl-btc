"""
tests/test_collect_top_of_book.py - Unit tests for the Phase B TOB collector.

Synthetic messages modeled on the venues' real (probed) schemas, no network. The
async I/O is verified by the live smoke run, not here; these tests pin the pure
parsers, symbol mapping, and the append-only daily-CSV sink.

Run:
    pytest tests/test_collect_top_of_book.py -v
"""

from __future__ import annotations

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
