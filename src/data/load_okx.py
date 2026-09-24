"""
src/data/load_okx.py - OKX perpetual funding-rate loader.

Parallel to `src/data/load_kraken.py`: fetches historical funding rates for an
OKX USDT-margined perpetual swap and returns them in a tidy DataFrame. Built for
the cross-venue funding-spread analysis, where OKX funding is compared against
Kraken funding.

THE CADENCE SUBTLETY (read before touching the numbers):
    OKX's published `fundingRate` is the rate for one SETTLEMENT PERIOD, not a
    per-hour rate. That period is NOT hourly and NOT fixed:
      - The historical default has been 4h, but the live BTC-USDT-SWAP cadence is
        currently 8h (verified against the API), and OKX dynamically escalates
        settlement frequency (e.g. 8h -> 4h -> 2h -> 1h) when the rate hits its
        cap/floor, reverting when the premium normalizes.
    Because the interval is dynamic, we DO NOT hardcode it. We INFER each row's
    interval from the gap to the previous funding timestamp, then normalize to a
    per-hour rate:

        funding_rate_hourly = funding_rate / interval_hours

    Kraken settles hourly, so its rate is already per-hour. Only after both venues
    are on this common per-hour basis is a funding spread between them meaningful.
    Subtracting an OKX period-rate from a Kraken hourly rate directly is a
    correctness bug that invalidates every downstream figure.

Endpoint (public, no auth):
    GET https://www.okx.com/api/v5/public/funding-rate-history?instId=...
    Paginated backwards via the `after` cursor (ms epoch), max 100 rows/request.

Public API:
    fetch_okx_funding_rates(inst_id, start, end)   fetch + normalize + cache

CLI smoke test:
    python -m src.data.load_okx
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Reuse Kraken loader's hardened HTTP helper (content-type check + retries) and
# time normalization rather than reimplementing them.
from src.data.load_kraken import (
    DEFAULT_CACHE_DIR,
    PER_REQUEST_DELAY_SEC,
    _http_get_json,
    _to_utc,
)

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

OKX_FUNDING_HISTORY_URL = "https://www.okx.com/api/v5/public/funding-rate-history"
DEFAULT_INST_ID = "BTC-USDT-SWAP"

# OKX caps a single response at 100 rows. Backwards pagination assembles history.
OKX_PAGE_LIMIT = 100

# Cap total pagination requests so a runaway loop can't hammer the API. At 100
# rows/request and the current 8h cadence, 30 pages ~= 3000 settlements ~= 2.7yr.
MAX_PAGE_REQUESTS = 30

# Last-resort interval when it cannot be inferred (e.g. a single-row series).
# The interval is normally INFERRED from timestamps; this is only a fallback.
DEFAULT_INTERVAL_HOURS = 4.0


# ============================================================
# HTTP fetch (paginated, backwards via `after` cursor)
# ============================================================

def _fetch_okx_page(inst_id: str, after: int | None) -> list[dict]:
    """Fetch one page of funding history, optionally older than `after` (ms epoch).

    OKX wraps results as {"code": "0", "msg": "", "data": [...]}. A non-"0" code
    is an API-level error even though the HTTP status may be 200, so we surface it
    explicitly with the venue's own message.
    """
    params: dict[str, str | int] = {"instId": inst_id, "limit": OKX_PAGE_LIMIT}
    if after is not None:
        params["after"] = after

    data = _http_get_json(OKX_FUNDING_HISTORY_URL, params=params)

    code = str(data.get("code", ""))
    if code != "0":
        raise ValueError(
            f"OKX API error for {inst_id}: code={code!r} msg={data.get('msg')!r}"
        )
    if "data" not in data:
        raise ValueError(f"Unexpected OKX response shape: keys={list(data.keys())}")
    return data["data"]


def _paginate_okx_funding(inst_id: str, start_ms: int | None) -> list[dict]:
    """Page backwards through OKX funding history until `start_ms` or the cap.

    Returns the raw concatenated row dicts (unsorted, possibly with duplicates at
    page boundaries; the caller normalizes). Logs and stops if MAX_PAGE_REQUESTS
    is hit, which means history is TRUNCATED, not exhausted.
    """
    rows: list[dict] = []
    after: int | None = None  # None -> newest page first
    for request_i in range(1, MAX_PAGE_REQUESTS + 1):
        page = _fetch_okx_page(inst_id, after)
        if not page:
            break  # no older data remains
        rows.extend(page)

        # `fundingTime` is ms epoch; the page is newest-first, so its last row is
        # the oldest. Use that as the next `after` cursor to walk further back.
        oldest_ms = min(int(r["fundingTime"]) for r in page)
        logger.info(
            "  OKX page %d: %d rows, oldest=%s",
            request_i, len(page),
            pd.to_datetime(oldest_ms, unit="ms", utc=True),
        )

        if start_ms is not None and oldest_ms <= start_ms:
            break  # walked back far enough to cover the requested start
        if len(page) < OKX_PAGE_LIMIT:
            break  # short page => reached the beginning of available history

        after = oldest_ms
        time.sleep(PER_REQUEST_DELAY_SEC)
    else:
        logger.warning(
            "OKX pagination hit MAX_PAGE_REQUESTS=%d; funding history is TRUNCATED "
            "(older data exists but was not fetched).",
            MAX_PAGE_REQUESTS,
        )
    return rows


# ============================================================
# Parsing / normalization
# ============================================================

def _okx_rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Convert raw OKX funding rows to a typed, per-hour-normalized DataFrame.

    Output columns:
        funding_rate         OKX's published PERIOD rate (float, as-is).
        interval_hours       settlement interval inferred from the gap to the
                             previous funding timestamp (captures dynamic cadence).
        funding_rate_hourly  funding_rate / interval_hours (per-hour normalized).
    """
    df = pd.DataFrame(rows)
    # OKX sends everything as strings; cast the two fields we use.
    df["funding_rate"] = df["fundingRate"].astype(float)  # period rate, string -> float
    df["timestamp"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]  # dedupe page-boundary overlaps

    # Infer each settlement's interval from the gap to the PREVIOUS timestamp.
    # The first row has no predecessor -> NaN; backfill it from the next interval
    # (a good local estimate) so we never pollute real cadence with the fallback.
    interval_hours = df.index.to_series().diff().dt.total_seconds() / 3600.0
    interval_hours = interval_hours.bfill()
    if interval_hours.isna().any():
        # Only possible with a single row: fall back to the documented default.
        interval_hours = interval_hours.fillna(DEFAULT_INTERVAL_HOURS)
    df["interval_hours"] = interval_hours.to_numpy()

    # Normalize the period rate to a per-hour rate so it shares Kraken's basis.
    #   OKX period rate / interval_hours = per-hour rate
    df["funding_rate_hourly"] = df["funding_rate"] / df["interval_hours"]

    return df[["funding_rate", "interval_hours", "funding_rate_hourly"]]


# ============================================================
# Caching
# ============================================================

def _okx_cache_path(inst_id: str, start_label: str, end_label: str, cache_dir: Path) -> Path:
    """Deterministic cache filename for an OKX funding request."""
    return cache_dir / f"okx_funding_{inst_id}_{start_label}_{end_label}.parquet"


# ============================================================
# Public entrypoint
# ============================================================

def fetch_okx_funding_rates(
    inst_id: str = DEFAULT_INST_ID,
    start: str | None = None,
    end: str | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch OKX historical funding rates, normalized to a per-hour basis.

    Args:
        inst_id: OKX instrument id, e.g. "BTC-USDT-SWAP" (USDT-margined BTC perp).
        start, end: optional ISO date strings ("2024-01-01") bounding the range.
            `start` limits how far back pagination walks; `end` trims the tail.
            Both interpreted as UTC. If omitted, fetch as much as the request cap
            allows (see MAX_PAGE_REQUESTS).
        cache_dir: directory for the Parquet cache (default data/raw).
        force_refresh: if True, refetch even if a cache file exists.

    Returns:
        DataFrame indexed by UTC funding timestamp with columns:
            funding_rate         OKX period rate (float, as published)
            interval_hours       inferred settlement interval (dynamic cadence)
            funding_rate_hourly  funding_rate / interval_hours (per-hour)

    Raises:
        ValueError: on an OKX API error code or an empty result.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start_ms = int(_to_utc(start).timestamp() * 1000) if start else None
    end_dt = _to_utc(end) if end else None

    start_label = start if start else "min"
    end_label = end if end else "max"
    cache_file = _okx_cache_path(inst_id, start_label, end_label, cache_dir)

    if cache_file.exists() and not force_refresh:
        logger.info("Loading cached OKX funding from %s", cache_file)
        return pd.read_parquet(cache_file)

    logger.info("Fetching OKX funding history: %s", inst_id)
    rows = _paginate_okx_funding(inst_id, start_ms)
    if not rows:
        raise ValueError(f"No OKX funding rates returned for inst_id {inst_id!r}")

    df = _okx_rows_to_dataframe(rows)

    # Trim to the requested window (inclusive of start, exclusive of end mirrors
    # the Kraken loader's convention).
    if start:
        df = df.loc[df.index >= _to_utc(start)]
    if end_dt is not None:
        df = df.loc[df.index < end_dt]

    if df.empty:
        raise ValueError(
            f"OKX funding for {inst_id!r} is empty after trimming to "
            f"[{start_label}, {end_label}). Widen the range."
        )

    df.to_parquet(cache_file)
    logger.info("Cached %d OKX funding rows to %s", len(df), cache_file)
    return df


# ============================================================
# OHLCV candle fetching (for cross-venue price analysis)
# ============================================================

# History-candles serves older data AND recent bars (newest-first, paginated
# backward via the `after` cursor). Max 100 rows/request.
OKX_HISTORY_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"
OKX_CANDLE_LIMIT = 100

# Cap total candle-pagination requests. At 100 bars/request, 5m bars, 500 pages
# ~= 50k bars ~= 170 days - far beyond the Phase A 3-4 week target.
MAX_CANDLE_PAGES = 500


def _fetch_okx_candle_page(inst_id: str, bar: str, after: int | None) -> list[list]:
    """Fetch one page of candles, optionally older than `after` (ms epoch).

    OKX candle rows are lists: [ts, open, high, low, close, vol, volCcy,
    volCcyQuote, confirm] - all strings; `ts` is the bar-OPEN time in ms,
    `confirm` is "1" for a closed bar and "0" for the in-progress one.
    """
    params: dict[str, str | int] = {"instId": inst_id, "bar": bar, "limit": OKX_CANDLE_LIMIT}
    if after is not None:
        params["after"] = after
    data = _http_get_json(OKX_HISTORY_CANDLES_URL, params=params)
    code = str(data.get("code", ""))
    if code != "0":
        raise ValueError(
            f"OKX candles API error for {inst_id} {bar}: code={code!r} msg={data.get('msg')!r}"
        )
    if "data" not in data:
        raise ValueError(f"Unexpected OKX candles response shape: keys={list(data.keys())}")
    return data["data"]


def _paginate_okx_candles(inst_id: str, bar: str, start_ms: int | None) -> list[list]:
    """Page backwards through OKX candles until `start_ms` or the request cap."""
    rows: list[list] = []
    after: int | None = None  # None -> newest page first
    for page_i in range(1, MAX_CANDLE_PAGES + 1):
        page = _fetch_okx_candle_page(inst_id, bar, after)
        if not page:
            break
        rows.extend(page)
        oldest_ms = min(int(r[0]) for r in page)
        if start_ms is not None and oldest_ms <= start_ms:
            break
        if len(page) < OKX_CANDLE_LIMIT:
            break  # short page => reached the start of available history
        after = oldest_ms
        time.sleep(PER_REQUEST_DELAY_SEC)
    else:
        logger.warning(
            "OKX candle pagination hit MAX_CANDLE_PAGES=%d; history TRUNCATED.",
            MAX_CANDLE_PAGES,
        )
    return rows


def _okx_candles_to_dataframe(rows: list[list]) -> pd.DataFrame:
    """Convert raw OKX candle rows to a typed OHLCV DataFrame.

    Keeps only CLOSED bars (confirm == "1") so a still-forming last candle never
    contaminates the analysis. Indexed by bar-open UTC timestamp; columns
    [open, high, low, close, volume].
    """
    # OKX history-candles rows have 9 fields; confirm is the last. Keep closed bars.
    closed = [r for r in rows if len(r) < 9 or str(r[-1]) == "1"]
    cols = ["ts", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote", "confirm"]
    df = pd.DataFrame(closed, columns=cols[: len(closed[0])])
    df["timestamp"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["open", "high", "low", "close", "volume"]]


def _okx_candle_cache_path(
    inst_id: str, bar: str, start_label: str, end_label: str, cache_dir: Path
) -> Path:
    """Deterministic cache filename for an OKX candle request."""
    return cache_dir / f"okx_ohlcv_{inst_id}_{bar}_{start_label}_{end_label}.parquet"


def fetch_okx_ohlcv(
    inst_id: str,
    bar: str = "5m",
    start: str | None = None,
    end: str | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch OKX candles for `inst_id` at `bar` resolution over [start, end).

    Args:
        inst_id: OKX instrument id, e.g. "BTC-USDT-SWAP".
        bar: OKX bar size, e.g. "5m", "1H", "1D".
        start, end: optional ISO date strings (UTC). `start` bounds how far back
            pagination walks; `end` trims the tail.
        cache_dir: Parquet cache directory.
        force_refresh: refetch even if a cache file exists.

    Returns:
        DataFrame indexed by bar-open UTC timestamp with columns
        [open, high, low, close, volume]. Only closed bars are included.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start_ms = int(_to_utc(start).timestamp() * 1000) if start else None
    end_dt = _to_utc(end) if end else None
    start_label = start if start else "min"
    end_label = end if end else "max"
    cache_file = _okx_candle_cache_path(inst_id, bar, start_label, end_label, cache_dir)

    if cache_file.exists() and not force_refresh:
        logger.info("Loading cached OKX candles from %s", cache_file)
        return pd.read_parquet(cache_file)

    logger.info("Fetching OKX candles: %s %s", inst_id, bar)
    rows = _paginate_okx_candles(inst_id, bar, start_ms)
    if not rows:
        raise ValueError(f"No OKX candles returned for {inst_id!r} {bar!r}")

    df = _okx_candles_to_dataframe(rows)
    if start:
        df = df.loc[df.index >= _to_utc(start)]
    if end_dt is not None:
        df = df.loc[df.index < end_dt]
    if df.empty:
        raise ValueError(
            f"OKX candles for {inst_id!r} {bar!r} empty after trimming to "
            f"[{start_label}, {end_label}). Widen the range."
        )

    df.to_parquet(cache_file)
    logger.info("Cached %d OKX candles to %s", len(df), cache_file)
    return df


# ============================================================
# CLI smoke test
# ============================================================

def _smoke_test() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("OKX funding-rate loader smoke test")
    print("=" * 60)

    # Force a fresh fetch so the smoke test actually exercises the network path.
    df = fetch_okx_funding_rates(DEFAULT_INST_ID, force_refresh=True)

    print(f"\nFetched {len(df)} funding settlements")
    print(f"Range: {df.index[0]} -> {df.index[-1]}")
    print(f"Columns: {list(df.columns)}")

    # Cadence is the headline: show the distribution of inferred intervals.
    print("\nInferred settlement interval (hours) value counts:")
    print(df["interval_hours"].round(2).value_counts())

    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nLast 3 rows:")
    print(df.tail(3))

    print("\nPer-hour normalized funding rate stats:")
    print(df["funding_rate_hourly"].describe())
    print("\nSmoke test PASSED")


if __name__ == "__main__":
    _smoke_test()
