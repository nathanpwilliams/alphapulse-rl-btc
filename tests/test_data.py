"""
tests/test_data.py - Unit tests for src/data/load_kraken.py

Mocks Kraken API responses so tests are fast, deterministic, and offline.
The real loader's correctness on live data is verified by its CLI smoke test
(`python -m src.data.load_kraken`); these tests verify the loader's logic
in isolation from network and API quirks.

Test categories:
    - Helpers: timestamp conversion, parsing, cache path
    - Fetching: pagination handling, error responses
    - Alignment: funding rates correctly matched to bar timestamps
    - Caching: write-then-read round trip
    - Contract: output passes TradingEnv._validate_data()

Run:
    pytest tests/test_data.py -v
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

from src.data import load_kraken as lk  # noqa: E402
from src.env.trading_env import TradingEnv  # noqa: E402


# ============================================================
# Helpers: synthetic Kraken API response builders
# ============================================================

def make_candle(time_ms: int, price: float, volume: float = 1000.0) -> dict:
    """Build a synthetic Kraken candle dict matching the real API shape."""
    return {
        "time": time_ms,
        "open": f"{price:.5f}",
        "high": f"{price * 1.005:.5f}",
        "low": f"{price * 0.995:.5f}",
        "close": f"{price:.5f}",
        "volume": volume,
    }


def make_kraken_candle_response(start_dt: datetime, n_bars: int, bar_hours: float = 4.0) -> dict:
    """Build a full synthetic chart response with `n_bars` candles starting at `start_dt`."""
    candles = []
    for i in range(n_bars):
        bar_dt = start_dt + pd.Timedelta(hours=bar_hours * i)
        time_ms = int(bar_dt.timestamp() * 1000)
        # Slight upward drift so all prices are unique
        price = 40000.0 * (1.0 + 0.0001 * i)
        candles.append(make_candle(time_ms, price))
    return {"candles": candles, "more_candles": False}


def make_kraken_funding_response(start_dt: datetime, n_rates: int, rate: float = 1e-5) -> dict:
    """Build a synthetic funding rate response at 4h intervals."""
    rates = []
    for i in range(n_rates):
        ts = start_dt + pd.Timedelta(hours=4 * i)
        rates.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "fundingRate": rate * 1e-5,        # absolute rate; we don't use this
            "relativeFundingRate": rate,        # what the loader actually reads
        })
    return {"rates": rates}


# ============================================================
# Time conversion helpers
# ============================================================

class TestTimeConversion:

    def test_string_to_utc(self):
        result = lk._to_utc("2024-01-15")
        assert result.tzinfo is not None
        assert result.year == 2024 and result.month == 1 and result.day == 15

    def test_datetime_naive_assumed_utc(self):
        naive = datetime(2024, 1, 15, 12, 0)
        result = lk._to_utc(naive)
        assert result.tzinfo == timezone.utc

    def test_datetime_with_tz_preserved(self):
        aware = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
        result = lk._to_utc(aware)
        assert result == aware

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            lk._to_utc(12345)


# ============================================================
# Candle parsing
# ============================================================

class TestCandleParsing:

    def test_parses_kraken_candle_shape(self):
        candles = [
            make_candle(time_ms=1704067200000, price=40000.0),  # 2024-01-01 00:00 UTC
            make_candle(time_ms=1704081600000, price=40100.0),  # 2024-01-01 04:00 UTC
        ]
        df = lk._candles_to_dataframe(candles)

        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"
        assert len(df) == 2
        assert df["close"].iloc[0] == pytest.approx(40000.0)
        assert df["close"].iloc[1] == pytest.approx(40100.0)

    def test_duplicates_deduplicated(self):
        candles = [
            make_candle(time_ms=1704067200000, price=40000.0),
            make_candle(time_ms=1704067200000, price=40000.0),  # exact dup
        ]
        df = lk._candles_to_dataframe(candles)
        assert len(df) == 1

    def test_unsorted_input_gets_sorted(self):
        candles = [
            make_candle(time_ms=1704081600000, price=40100.0),  # later first
            make_candle(time_ms=1704067200000, price=40000.0),  # earlier second
        ]
        df = lk._candles_to_dataframe(candles)
        assert df.index.is_monotonic_increasing


# ============================================================
# Fetching (with mocked HTTP)
# ============================================================

class TestFetchOHLCV:

    def test_fetches_and_returns_dataframe(self):
        start_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        response = make_kraken_candle_response(start_dt, n_bars=10)

        with patch.object(lk, "_http_get_json", return_value=response):
            df = lk.fetch_ohlcv("PF_XBTUSD", "2024-01-01", "2024-01-02", "4h")

        assert len(df) == 10
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}

    def test_rejects_invalid_resolution(self):
        with pytest.raises(ValueError, match="Invalid resolution"):
            lk.fetch_ohlcv("PF_XBTUSD", "2024-01-01", "2024-01-02", "7h")

    def test_rejects_inverted_date_range(self):
        with pytest.raises(ValueError, match="must be after"):
            lk.fetch_ohlcv("PF_XBTUSD", "2024-01-02", "2024-01-01", "4h")

    def test_empty_response_raises(self):
        with patch.object(lk, "_http_get_json", return_value={"candles": [], "more_candles": False}):
            with pytest.raises(ValueError, match="No candles returned"):
                lk.fetch_ohlcv("PF_XBTUSD", "2024-01-01", "2024-01-02", "4h")

    def test_malformed_response_raises(self):
        with patch.object(lk, "_http_get_json", return_value={"oops": "wrong shape"}):
            with pytest.raises(ValueError, match="Unexpected response shape"):
                lk.fetch_ohlcv("PF_XBTUSD", "2024-01-01", "2024-01-02", "4h")


class TestFetchFunding:

    def test_fetches_and_returns_dataframe(self):
        start_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        response = make_kraken_funding_response(start_dt, n_rates=10, rate=1e-5)

        with patch.object(lk, "_http_get_json", return_value=response):
            df = lk.fetch_funding_rates("PF_XBTUSD")

        assert list(df.columns) == ["funding_rate"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"
        assert len(df) == 10
        assert df["funding_rate"].iloc[0] == pytest.approx(1e-5)

    def test_empty_funding_response_raises(self):
        with patch.object(lk, "_http_get_json", return_value={"rates": []}):
            with pytest.raises(ValueError, match="No funding rates returned"):
                lk.fetch_funding_rates("PF_XBTUSD")


# ============================================================
# Funding alignment - the most important correctness property
# ============================================================

class TestFundingAlignment:
    """Funding rates published at 4-hour intervals must merge onto 4H bars
    such that each bar gets its corresponding rate, and the merge does not
    fabricate, lose, or shift any bars."""

    def _make_bars(self, n_bars: int, start: str = "2024-01-01") -> pd.DataFrame:
        """Build minimal bars DataFrame at 4H spacing."""
        idx = pd.date_range(start, periods=n_bars, freq="4h", tz="UTC")
        return pd.DataFrame(
            {
                "open": np.full(n_bars, 40000.0),
                "high": np.full(n_bars, 40000.0),
                "low": np.full(n_bars, 40000.0),
                "close": np.full(n_bars, 40000.0),
                "volume": np.full(n_bars, 1000.0),
            },
            index=idx,
        )

    def _make_funding(self, n_rates: int, rate: float = 1e-5, start: str = "2024-01-01") -> pd.DataFrame:
        idx = pd.date_range(start, periods=n_rates, freq="4h", tz="UTC")
        return pd.DataFrame({"funding_rate": np.full(n_rates, rate)}, index=idx)

    def test_perfect_alignment_preserves_all_bars(self):
        bars = self._make_bars(n_bars=50)
        funding = self._make_funding(n_rates=50, rate=1e-5)
        merged = lk.align_funding_to_bars(bars, funding, bar_hours=4.0)

        assert len(merged) == len(bars), "Merge changed row count"
        assert "funding_rate" in merged.columns
        assert (merged["funding_rate"] == 1e-5).all()

    def test_alignment_no_lookahead(self):
        """A funding rate published at 04:00 must apply to the bar opening at 04:00,
        not the bar that opened at 00:00 (which would be using future information)."""
        bars = self._make_bars(n_bars=3)  # 00:00, 04:00, 08:00

        # Funding rate only at 04:00 with distinct value
        funding = pd.DataFrame(
            {"funding_rate": [0.0001]},
            index=pd.DatetimeIndex(["2024-01-01 04:00:00"], tz="UTC"),
        )
        merged = lk.align_funding_to_bars(bars, funding, bar_hours=4.0)

        # Bar at 00:00 should NOT have the 04:00 rate (would be look-ahead)
        assert merged["funding_rate"].iloc[0] == 0.0, "Bar at 00:00 was given future funding rate (look-ahead!)"
        # Bar at 04:00 SHOULD have the 04:00 rate
        assert merged["funding_rate"].iloc[1] == pytest.approx(0.0001)

    def test_missing_funding_filled_with_zero(self):
        """If funding history is too short to cover all bars, fill with 0 and warn."""
        bars = self._make_bars(n_bars=10)
        funding = self._make_funding(n_rates=5)  # only first half covered
        merged = lk.align_funding_to_bars(bars, funding, bar_hours=4.0)

        assert len(merged) == 10
        assert (merged["funding_rate"].iloc[5:] == 0.0).all()
        assert not merged["funding_rate"].isna().any()

    def test_alignment_rejects_non_datetime_index(self):
        bars = self._make_bars(n_bars=5).reset_index(drop=True)
        funding = self._make_funding(n_rates=5)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            lk.align_funding_to_bars(bars, funding)


# ============================================================
# Caching round-trip
# ============================================================

class TestCaching:

    def test_cache_path_deterministic(self):
        p1 = lk._cache_path(
            "PF_XBTUSD",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 4, 1, tzinfo=timezone.utc),
            "4h",
            Path("/tmp"),
        )
        p2 = lk._cache_path(
            "PF_XBTUSD",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 4, 1, tzinfo=timezone.utc),
            "4h",
            Path("/tmp"),
        )
        assert p1 == p2
        assert str(p1).endswith(".parquet")

    def test_write_then_read_roundtrip(self, tmp_path):
        idx = pd.date_range("2024-01-01", periods=100, freq="4h", tz="UTC")
        df = pd.DataFrame(
            {
                "open": np.linspace(40000, 41000, 100),
                "high": np.linspace(40100, 41100, 100),
                "low": np.linspace(39900, 40900, 100),
                "close": np.linspace(40000, 41000, 100),
                "volume": np.full(100, 1000.0),
                "funding_rate": np.full(100, 1e-5),
            },
            index=idx,
        )
        cache_file = tmp_path / "test.parquet"
        df.to_parquet(cache_file)

        loaded = lk.load_cached(cache_file)
        pd.testing.assert_frame_equal(df, loaded, check_freq=False)

    def test_load_cached_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            lk.load_cached("/nonexistent/path.parquet")


# ============================================================
# Contract with TradingEnv
# ============================================================

class TestEnvContract:
    """The loader is BUILT to satisfy TradingEnv's data validation contract.
    Any output it produces must pass into TradingEnv without errors."""

    def test_loader_output_passes_env_validation(self, tmp_path):
        """End-to-end: mock both API responses, build dataset, feed to env."""
        # Need at least 1000 bars to satisfy env warmup + episode requirements
        start_dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
        n_bars = 1200

        candle_response = make_kraken_candle_response(start_dt, n_bars=n_bars)
        funding_response = make_kraken_funding_response(start_dt, n_rates=n_bars)

        # Patch both endpoints; the loader's _http_get_json is called for each
        def fake_http(url, params=None):
            if "charts/v1" in url:
                return candle_response
            elif "funding" in url:
                return funding_response
            raise AssertionError(f"Unexpected URL: {url}")

        with patch.object(lk, "_http_get_json", side_effect=fake_http):
            df = lk.build_dataset(
                symbol="PF_XBTUSD",
                start="2023-01-01",
                end="2023-09-01",  # Wide enough to capture most of the 1200 bars
                cache_dir=tmp_path,
            )

        # The env should accept this DataFrame without raising
        env = TradingEnv(data=df)
        obs, info = env.reset(seed=0)
        assert obs is not None
        assert env.observation_space.shape == (24,)
