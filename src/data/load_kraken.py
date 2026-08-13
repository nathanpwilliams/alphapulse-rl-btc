"""
src/data/load_kraken.py - Kraken Futures historical data loader.

Fetches 4H OHLCV bars and historical funding rates for a Kraken Futures
perpetual contract, aligns them on timestamp, and returns a DataFrame in
exactly the schema TradingEnv requires.

Public API:
    build_dataset(symbol, start, end, ...)    main entrypoint; fetches + caches
    load_cached(path)                          fast read of an existing Parquet

Design principles:
    DETERMINISTIC. Given the same (symbol, start, end), result is identical.
    CACHED. First call hits Kraken; subsequent calls read Parquet from disk.
    CONTRACT-CHECKED. Output passes TradingEnv._validate_data() by construction.
    NO AUTH NEEDED. Both endpoints are public (no API key, no signing).
    RESPECTFUL TO API. Paginated requests with small delays; explicit retries.

References:
    OHLC:    https://futures.kraken.com/api/charts/v1/{tick_type}/{symbol}/{resolution}
    Funding: https://futures.kraken.com/api/v4/historicalfundingrates?symbol={symbol}

Usage:
    from src.data.load_kraken import build_dataset
    df = build_dataset(symbol="PF_XBTUSD", start="2022-01-01", end="2026-01-01")
    # df is ready to feed directly to TradingEnv

CLI smoke test:
    python -m src.data.load_kraken
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


# ============================================================
# Constants
# ============================================================

KRAKEN_FUTURES_BASE = "https://futures.kraken.com/api"
DEFAULT_SYMBOL = "PF_XBTUSD"
DEFAULT_RESOLUTION = "4h"
DEFAULT_CACHE_DIR = Path("data/raw")

# Conservative pacing for Kraken's public API rate limits
PER_REQUEST_DELAY_SEC = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0

# Valid resolutions per Kraken docs
VALID_RESOLUTIONS = {"1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d", "1w"}

# Approximate bar-hours, used for chunk math and validation
RESOLUTION_HOURS = {
    "1m": 1 / 60, "5m": 5 / 60, "15m": 15 / 60, "30m": 30 / 60,
    "1h": 1.0, "4h": 4.0, "12h": 12.0, "1d": 24.0, "1w": 168.0,
}

logger = logging.getLogger(__name__)


# ============================================================
# HTTP helpers (with retry)
# ============================================================

def _http_get_json(url: str, params: dict | None = None) -> dict:
    """GET a URL, parse JSON, retry on transient failures.

    On non-JSON responses (HTML error pages, empty bodies), raises a clear
    ValueError that names the actual problem instead of leaking a generic
    JSONDecodeError.
    """
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "json" not in content_type.lower():
                raise ValueError(
                    f"Expected JSON from {url}, got {content_type!r} "
                    f"(status {resp.status_code}, {len(resp.text)} chars). "
                    f"First 200 chars: {resp.text[:200]!r}"
                )
            return resp.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SEC * attempt
                logger.warning(
                    "Request to %s failed (attempt %d/%d): %s. Retrying in %.1fs",
                    url, attempt, MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")


# ============================================================
# OHLC fetching (paginated)
# ============================================================

def _fetch_ohlcv_chunk(
    symbol: str,
    from_ts: int,
    to_ts: int,
    resolution: str,
    tick_type: str = "trade",
) -> tuple[list[dict], bool]:
    """Fetch one chunk of candles from Kraken charts endpoint.

    Args:
        symbol: e.g., "PF_XBTUSD"
        from_ts, to_ts: Unix timestamps in seconds (inclusive range)
        resolution: one of VALID_RESOLUTIONS
        tick_type: "trade" (default), "mark", or "spot"

    Returns:
        (candles_list, more_candles_flag)
    """
    url = f"{KRAKEN_FUTURES_BASE}/charts/v1/{tick_type}/{symbol}/{resolution}"
    params = {"from": from_ts, "to": to_ts}
    data = _http_get_json(url, params=params)

    if "candles" not in data:
        raise ValueError(f"Unexpected response shape from {url}: keys={list(data.keys())}")

    return data["candles"], bool(data.get("more_candles", False))


def fetch_ohlcv(
    symbol: str,
    start: str | datetime,
    end: str | datetime,
    resolution: str = DEFAULT_RESOLUTION,
) -> pd.DataFrame:
    """Fetch OHLCV bars from Kraken Futures over the given date range.

    Paginates by following Kraken's `more_candles` flag rather than guessing a
    chunk size. Kraken caps each charts response at ~2000 candles and sets
    `more_candles=True` when it truncates; we advance a `from` cursor to just past
    the last returned candle and refetch until the API reports no more candles or
    we reach `end`. This is resolution-agnostic: a truncated 4H window and a
    truncated 1H window are handled identically, with no silently dropped tail.

    Args:
        symbol: e.g., "PF_XBTUSD"
        start, end: ISO date strings ("2022-01-01") or datetime objects.
                    Interpreted as UTC.
        resolution: one of VALID_RESOLUTIONS (default "4h")

    Returns:
        DataFrame indexed by UTC datetime, columns [open, high, low, close, volume].
        All numeric columns are float64.
    """
    if resolution not in VALID_RESOLUTIONS:
        raise ValueError(
            f"Invalid resolution {resolution!r}. Valid: {sorted(VALID_RESOLUTIONS)}"
        )

    start_dt = _to_utc(start)
    end_dt = _to_utc(end)
    if end_dt <= start_dt:
        raise ValueError(f"end ({end_dt}) must be after start ({start_dt})")

    logger.info("Fetching OHLCV: %s %s from %s to %s", symbol, resolution, start_dt, end_dt)

    # Cursor-driven pagination. `from`/`to` are Unix SECONDS (Kraken's charts
    # params); candle `time` is Unix MILLISECONDS. The requested end is fixed;
    # only the `from` cursor moves forward as we page.
    all_candles: list[dict] = []
    to_ts = int(end_dt.timestamp())
    cursor_ts = int(start_dt.timestamp())

    # Belt-and-suspenders cap on total requests. The cursor-advance check below is
    # the real termination guard; MAX_PAGES only bounds a pathological API. At
    # ~2000 candles/page this is far more than any realistic multi-year request.
    MAX_PAGES = 10_000

    for page_i in range(1, MAX_PAGES + 1):
        candles, more = _fetch_ohlcv_chunk(symbol, cursor_ts, to_ts, resolution)
        if not candles:
            # No candles at/after the cursor within range: history is exhausted.
            break
        all_candles.extend(candles)

        # Last candle's bar-start time (ms -> s). Dedupe in _candles_to_dataframe
        # is the safety net for any boundary overlap.
        last_ms = max(int(c["time"]) for c in candles)
        last_s = last_ms // 1000
        logger.info(
            "  Page %d: %d candles up to %s (more_candles=%s)",
            page_i, len(candles),
            pd.to_datetime(last_ms, unit="ms", utc=True), more,
        )

        # Done when the API says there's no more, or we've reached the end.
        if not more or last_s >= to_ts:
            break

        # Advance to just past the last candle: the next page starts at the
        # following bar (no gap) without re-requesting the boundary bar (no dup).
        next_cursor_ts = last_s + 1
        if next_cursor_ts <= cursor_ts:
            # Cursor failed to move forward (API returned only stale/older bars):
            # stop rather than spin. Data may be incomplete; say so.
            logger.warning(
                "OHLCV pagination cursor did not advance (last candle %s <= cursor "
                "%s); stopping to avoid an infinite loop. Data may be incomplete.",
                pd.to_datetime(last_ms, unit="ms", utc=True),
                pd.to_datetime(cursor_ts, unit="s", utc=True),
            )
            break
        cursor_ts = next_cursor_ts
        time.sleep(PER_REQUEST_DELAY_SEC)
    else:
        logger.warning(
            "OHLCV pagination hit MAX_PAGES=%d; assembled history may be truncated.",
            MAX_PAGES,
        )

    if not all_candles:
        raise ValueError(
            f"No candles returned for {symbol} {resolution} {start_dt}..{end_dt}. "
            f"Check the symbol exists and the date range covers a trading period."
        )

    return _candles_to_dataframe(all_candles)


def _candles_to_dataframe(candles: list[dict]) -> pd.DataFrame:
    """Convert Kraken's candle JSON list to a typed DataFrame."""
    df = pd.DataFrame(candles)
    # Kraken returns prices as strings, volume as a number; normalize to float
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float)

    # `time` is Unix milliseconds at bar START
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]

    # Deduplicate by index in case overlapping chunks returned the same bar
    df = df[~df.index.duplicated(keep="first")]
    return df


# ============================================================
# Funding rate fetching
# ============================================================

def fetch_funding_rates(symbol: str) -> pd.DataFrame:
    """Fetch all historical funding rates for a symbol.

    Kraken returns the full history in one response; we filter client-side
    when we merge into the dataset.

    Returns:
        DataFrame indexed by UTC datetime, single column `funding_rate`
        (the relativeFundingRate — proportional rate for the funding period).
    """
    url = "https://futures.kraken.com/derivatives/api/v3/historical-funding-rates"
    logger.info("Fetching funding rates: %s", symbol)
    data = _http_get_json(url, params={"symbol": symbol})

    if "rates" not in data:
        raise ValueError(f"Unexpected response shape from {url}: keys={list(data.keys())}")

    rates = data["rates"]
    if not rates:
        raise ValueError(f"No funding rates returned for symbol {symbol!r}")

    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.rename(columns={"relativeFundingRate": "funding_rate"})
    df = df.set_index("timestamp").sort_index()
    df = df[["funding_rate"]].astype(float)
    df = df[~df.index.duplicated(keep="first")]
    return df


# ============================================================
# Alignment
# ============================================================

def align_funding_to_bars(
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    bar_hours: float = 4.0,
) -> pd.DataFrame:
    """Merge funding rates onto OHLCV bars by timestamp.

    Kraken publishes funding hourly. For each 4H bar, we attach the funding
    rate whose timestamp falls within a tight tolerance of the bar's open.
    Bars with no matching funding rate within tolerance fall through to the
    NaN -> 0 fill path; a near-total miss (>95%) raises, because that almost
    always means the requested date range is outside the funding history window.

    Args:
        bars: DataFrame with DatetimeIndex and OHLCV columns
        funding: DataFrame with DatetimeIndex and `funding_rate` column
        bar_hours: bar duration in hours (default 4.0, unused for current
                   tolerance but kept for API stability)

    Returns:
        DataFrame with all OHLCV columns + funding_rate, indexed identically to bars.
    """
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("`bars` must have a DatetimeIndex")
    if not isinstance(funding.index, pd.DatetimeIndex):
        raise ValueError("`funding` must have a DatetimeIndex")

    # Tolerance handles only minor timestamp jitter from Kraken's API.
    # Kept well below bar_hours so a missing funding rate cannot be silently
    # substituted with an older, stale rate.
    tolerance = pd.Timedelta(minutes=30)
    bars_sorted = bars.sort_index()
    funding_sorted = funding.sort_index()

    merged = pd.merge_asof(
        bars_sorted,
        funding_sorted,
        left_index=True,
        right_index=True,
        direction="backward",
        tolerance=tolerance,
    )

    n_missing = merged["funding_rate"].isna().sum()
    if n_missing > 0:
        pct = 100.0 * n_missing / len(merged)
        if pct > 95.0:
            raise ValueError(
                f"Funding alignment failed: {n_missing}/{len(merged)} bars "
                f"({pct:.1f}%) had no matching funding rate. This usually means "
                f"the requested date range is outside the funding history window "
                f"(Kraken's funding endpoint typically returns ~1 year). "
                f"Funding history begins at {funding.index[0]}, requested bars span "
                f"{bars.index[0]} to {bars.index[-1]}."
            )
        logger.warning(
            "Funding alignment: %d/%d bars (%.1f%%) had no matching rate. "
            "Filling with 0.0.",
            n_missing, len(merged), pct,
        )
        merged["funding_rate"] = merged["funding_rate"].fillna(0.0)

    return merged


# ============================================================
# Caching
# ============================================================

def _cache_path(
    symbol: str,
    start: datetime,
    end: datetime,
    resolution: str,
    cache_dir: Path,
) -> Path:
    """Deterministic cache filename based on the request parameters."""
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    fname = f"{symbol}_{resolution}_{start_str}_{end_str}.parquet"
    return cache_dir / fname


def load_cached(path: Path | str) -> pd.DataFrame:
    """Read a cached Parquet file, with light validation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No cache at {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Cached file {path} has wrong index type")
    return df


# ============================================================
# Main entrypoint
# ============================================================

def build_dataset(
    symbol: str = DEFAULT_SYMBOL,
    start: str = "2022-01-01",
    end: str | None = None,
    resolution: str = DEFAULT_RESOLUTION,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Build a complete, env-ready dataset by fetching + merging + caching.

    On first call: hits Kraken API, builds the dataset, caches to Parquet, returns it.
    On subsequent calls with the same args: reads the Parquet cache directly.

    Args:
        symbol: Kraken Futures symbol (e.g., "PF_XBTUSD")
        start, end: ISO date strings. `end` defaults to today (UTC).
        resolution: bar resolution (default "4h")
        cache_dir: where to store the Parquet cache
        force_refresh: if True, refetch even if cache exists

    Returns:
        DataFrame indexed by UTC datetime with columns:
            [open, high, low, close, volume, funding_rate]
        This is exactly the schema TradingEnv expects.
    """
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_dt = _to_utc(start)
    end_dt = _to_utc(end)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = _cache_path(symbol, start_dt, end_dt, resolution, cache_dir)

    if cache_file.exists() and not force_refresh:
        logger.info("Loading cached dataset from %s", cache_file)
        return load_cached(cache_file)

    # Fresh build
    logger.info("Building dataset from scratch (cache miss or force_refresh=True)")
    bars = fetch_ohlcv(symbol, start_dt, end_dt, resolution)
    funding = fetch_funding_rates(symbol)

    bar_h = RESOLUTION_HOURS[resolution]
    merged = align_funding_to_bars(bars, funding, bar_hours=bar_h)

    # Trim funding's full history down to the requested window
    merged = merged.loc[(merged.index >= start_dt) & (merged.index < end_dt)]

    # Final sanity checks before writing cache
    _validate_output(merged)

    # Persist
    merged.to_parquet(cache_file)
    logger.info("Cached %d bars to %s", len(merged), cache_file)

    return merged


def _validate_output(df: pd.DataFrame) -> None:
    """Ensure the merged dataset matches the env's expectations."""
    required = {"open", "high", "low", "close", "volume", "funding_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Output missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Output index must be DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Output index must be monotonic increasing")
    if df["close"].isna().any() or (df["close"] <= 0).any():
        raise ValueError("close must be strictly positive")
    if df["funding_rate"].isna().any():
        raise ValueError("funding_rate has NaN values (alignment failed)")


# ============================================================
# Time conversion
# ============================================================

def _to_utc(t: str | datetime | pd.Timestamp) -> datetime:
    """Normalize various date inputs to a tz-aware UTC datetime."""
    if isinstance(t, str):
        return pd.Timestamp(t, tz="UTC").to_pydatetime()
    if isinstance(t, pd.Timestamp):
        if t.tz is None:
            return t.tz_localize("UTC").to_pydatetime()
        return t.tz_convert("UTC").to_pydatetime()
    if isinstance(t, datetime):
        if t.tzinfo is None:
            return t.replace(tzinfo=timezone.utc)
        return t.astimezone(timezone.utc)
    raise TypeError(f"Cannot convert {type(t).__name__} to UTC datetime")


# ============================================================
# CLI smoke test
# ============================================================

def _smoke_test():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("Kraken Futures data loader smoke test")
    print("=" * 60)

    # Small window so the test runs quickly
    df = build_dataset(
        symbol=DEFAULT_SYMBOL,
        start="2026-03-01",
        end="2026-06-01",
        resolution="4h",
    )

    print(f"\nFetched {len(df)} bars")
    print(f"Range: {df.index[0]} -> {df.index[-1]}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    print(f"\nLast 3 rows:")
    print(df.tail(3))
    print(f"\nFunding rate stats:")
    print(df["funding_rate"].describe())
    print("\nSmoke test PASSED")


if __name__ == "__main__":
    _smoke_test()
