"""
tests/test_load_okx.py - Unit tests for src/data/load_okx.py

Mocks the OKX API so tests are fast, deterministic, and offline (mirrors the
approach in tests/test_data.py for the Kraken loader). The key correctness
property is the cadence normalization: OKX publishes a PERIOD rate on a dynamic
interval, and the loader must infer that interval from timestamps and normalize
to a per-hour rate.

Run:
    pytest tests/test_load_okx.py -v
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data import load_okx as lo  # noqa: E402


# ============================================================
# Synthetic OKX response builders
# ============================================================

def make_okx_rows(start_dt: datetime, n: int, rate: float = 1e-4, interval_h: float = 8.0) -> list[dict]:
    """Build synthetic OKX funding rows (newest-first, as the API returns them)."""
    rows = []
    for i in range(n):
        ts = start_dt + pd.Timedelta(hours=interval_h * i)
        rows.append({
            "instId": "BTC-USDT-SWAP",
            "instType": "SWAP",
            "fundingRate": f"{rate:.12f}",     # OKX sends the period rate as a string
            "realizedRate": f"{rate:.12f}",
            "fundingTime": str(int(ts.timestamp() * 1000)),
        })
    # API returns newest-first; reverse so index 0 is the most recent.
    return list(reversed(rows))


# ============================================================
# Parsing / normalization
# ============================================================

class TestOkxParsing:

    def test_parses_shape_and_types(self):
        rows = make_okx_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), n=10, rate=1e-4, interval_h=8.0)
        df = lo._okx_rows_to_dataframe(rows)

        assert list(df.columns) == ["funding_rate", "interval_hours", "funding_rate_hourly"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"
        assert len(df) == 10
        assert df["funding_rate"].dtype == float
        assert df.index.is_monotonic_increasing

    def test_funding_rate_cast_from_string(self):
        rows = make_okx_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), n=3, rate=1.23e-4)
        df = lo._okx_rows_to_dataframe(rows)
        assert df["funding_rate"].iloc[0] == pytest.approx(1.23e-4)

    def test_interval_inferred_from_timestamps(self):
        # 8-hourly settlements => inferred interval should be 8.0 everywhere.
        rows = make_okx_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), n=12, interval_h=8.0)
        df = lo._okx_rows_to_dataframe(rows)
        assert np.allclose(df["interval_hours"], 8.0)

    def test_interval_inferred_4h(self):
        rows = make_okx_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), n=12, interval_h=4.0)
        df = lo._okx_rows_to_dataframe(rows)
        assert np.allclose(df["interval_hours"], 4.0)

    def test_hourly_normalization_divides_by_interval(self):
        # A 4h-period rate must normalize to rate / 4 per hour.
        rate = 4e-4
        rows = make_okx_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), n=6, rate=rate, interval_h=4.0)
        df = lo._okx_rows_to_dataframe(rows)
        assert np.allclose(df["funding_rate_hourly"], rate / 4.0)

    def test_hourly_normalization_8h(self):
        rate = 8e-4
        rows = make_okx_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), n=6, rate=rate, interval_h=8.0)
        df = lo._okx_rows_to_dataframe(rows)
        # 8h period rate / 8 = per-hour rate
        assert np.allclose(df["funding_rate_hourly"], rate / 8.0)

    def test_dynamic_interval_mixed_cadence(self):
        """When cadence changes (8h then 4h), each row's interval reflects the gap
        to its own predecessor, and normalization follows suit."""
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Build ascending timestamps: 0h, 8h, 16h (8h cadence), then 20h, 24h (4h).
        offsets_h = [0, 8, 16, 20, 24]
        rows = []
        for off in offsets_h:
            ts = base + pd.Timedelta(hours=off)
            rows.append({
                "instId": "BTC-USDT-SWAP",
                "fundingRate": "0.0001",
                "fundingTime": str(int(ts.timestamp() * 1000)),
            })
        df = lo._okx_rows_to_dataframe(list(reversed(rows)))
        intervals = df["interval_hours"].to_numpy()
        # First row backfilled from the next gap (8h); then 8, 8, 4, 4.
        assert intervals == pytest.approx([8.0, 8.0, 8.0, 4.0, 4.0])
        # Same 0.0001 period rate normalizes differently under 8h vs 4h.
        hourly = df["funding_rate_hourly"].to_numpy()
        assert hourly[2] == pytest.approx(0.0001 / 8.0)
        assert hourly[4] == pytest.approx(0.0001 / 4.0)

    def test_duplicates_deduplicated(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = make_okx_rows(base, n=3, interval_h=8.0)
        rows = rows + rows  # duplicate every row
        df = lo._okx_rows_to_dataframe(rows)
        assert len(df) == 3


# ============================================================
# Pagination / fetch (mocked HTTP)
# ============================================================

class TestOkxFetch:

    def test_api_error_code_raises(self):
        with patch.object(lo, "_http_get_json", return_value={"code": "50011", "msg": "rate limited", "data": []}):
            with pytest.raises(ValueError, match="OKX API error"):
                lo._fetch_okx_page("BTC-USDT-SWAP", after=None)

    def test_missing_data_key_raises(self):
        with patch.object(lo, "_http_get_json", return_value={"code": "0", "msg": ""}):
            with pytest.raises(ValueError, match="Unexpected OKX response"):
                lo._fetch_okx_page("BTC-USDT-SWAP", after=None)

    def test_fetch_paginates_and_normalizes(self, tmp_path):
        """A short first page ends pagination; result is normalized + cached."""
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = make_okx_rows(base, n=50, rate=8e-4, interval_h=8.0)  # < 100 => single page

        with patch.object(lo, "_http_get_json", return_value={"code": "0", "msg": "", "data": rows}):
            df = lo.fetch_okx_funding_rates("BTC-USDT-SWAP", cache_dir=tmp_path)

        assert len(df) == 50
        assert np.allclose(df["funding_rate_hourly"], 8e-4 / 8.0)
        # Cache file should now exist.
        assert any(tmp_path.glob("okx_funding_*.parquet"))

    def test_cache_roundtrip(self, tmp_path):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = make_okx_rows(base, n=20, interval_h=8.0)

        with patch.object(lo, "_http_get_json", return_value={"code": "0", "msg": "", "data": rows}) as m:
            df1 = lo.fetch_okx_funding_rates("BTC-USDT-SWAP", cache_dir=tmp_path)
            calls_after_first = m.call_count
            # Second call should read cache, not hit HTTP again.
            df2 = lo.fetch_okx_funding_rates("BTC-USDT-SWAP", cache_dir=tmp_path)
            assert m.call_count == calls_after_first

        pd.testing.assert_frame_equal(df1, df2)

    def test_empty_result_raises(self, tmp_path):
        with patch.object(lo, "_http_get_json", return_value={"code": "0", "msg": "", "data": []}):
            with pytest.raises(ValueError, match="No OKX funding"):
                lo.fetch_okx_funding_rates("BTC-USDT-SWAP", cache_dir=tmp_path)


# ============================================================
# OHLCV candle fetching
# ============================================================

def make_okx_candles(start_dt: datetime, n: int, price: float = 40000.0,
                     bar_min: int = 5, confirm: str = "1") -> list[list]:
    """Synthetic OKX candle rows (newest-first, as history-candles returns them)."""
    rows = []
    for i in range(n):
        ts = start_dt + pd.Timedelta(minutes=bar_min * i)
        p = price * (1.0 + 0.0001 * i)
        rows.append([
            str(int(ts.timestamp() * 1000)),
            f"{p:.2f}", f"{p * 1.001:.2f}", f"{p * 0.999:.2f}", f"{p:.2f}",
            "10.0", "400000.0", "400000.0", confirm,
        ])
    return list(reversed(rows))  # API returns newest-first


class TestOkxCandles:

    def test_parses_candle_shape(self):
        rows = make_okx_candles(datetime(2024, 1, 1, tzinfo=timezone.utc), n=10)
        df = lo._okx_candles_to_dataframe(rows)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"
        assert len(df) == 10
        assert df.index.is_monotonic_increasing
        assert df["close"].iloc[0] == pytest.approx(40000.0)

    def test_unconfirmed_bars_dropped(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        closed = make_okx_candles(base, n=5)
        # Prepend an in-progress (confirm="0") most-recent bar.
        live = make_okx_candles(base + pd.Timedelta(minutes=25), n=1, confirm="0")
        df = lo._okx_candles_to_dataframe(live + closed)
        assert len(df) == 5  # the confirm="0" bar is excluded

    def test_fetch_paginates_and_caches(self, tmp_path):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = make_okx_candles(base, n=50, bar_min=5)  # < 100 -> single page

        with patch.object(lo, "_http_get_json", return_value={"code": "0", "msg": "", "data": rows}):
            df = lo.fetch_okx_ohlcv("BTC-USDT-SWAP", bar="5m", cache_dir=tmp_path)

        assert len(df) == 50
        assert any(tmp_path.glob("okx_ohlcv_*.parquet"))

    def test_candle_api_error_raises(self):
        with patch.object(lo, "_http_get_json", return_value={"code": "50011", "msg": "x", "data": []}):
            with pytest.raises(ValueError, match="OKX candles API error"):
                lo._fetch_okx_candle_page("BTC-USDT-SWAP", "5m", after=None)

    def test_candle_pagination_walks_backward(self, tmp_path):
        """Two full pages then a short page: cursor advances via `after`."""
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        page1 = make_okx_candles(base + pd.Timedelta(minutes=5 * 100), n=100)  # newest
        page2 = make_okx_candles(base, n=40)  # older, short -> ends pagination

        calls = {"n": 0}
        def fake(url, params=None):
            calls["n"] += 1
            return {"code": "0", "msg": "", "data": page1 if calls["n"] == 1 else page2}

        with patch.object(lo, "_http_get_json", side_effect=fake):
            df = lo.fetch_okx_ohlcv("BTC-USDT-SWAP", bar="5m", cache_dir=tmp_path)
        assert calls["n"] == 2
        assert len(df) == 140
        assert df.index.is_monotonic_increasing
