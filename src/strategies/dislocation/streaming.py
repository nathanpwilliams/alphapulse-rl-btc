"""
src/strategies/dislocation/streaming.py - Chunked multi-day driver for the
executable-gap analyzer (Phase B).

The Phase B collection is ~71M rows / ~4.3 GB of daily top-of-book CSVs, far too
large to concatenate into one frame (which is what `executable.load_tob` does).
This module drives the SAME analyzer math one (daily file x asset) slice at a
time, keeping only per-excursion summary records in memory:

    for each daily tob_YYYYMMDD.csv:
        for each asset:
            read the file in chunks, keep only that asset's rows  <- bounded memory
            build_executable_series(...)                          <- reused as-is
            extract one record per net-positive excursion          <- small
            discard the slice

Nothing here re-implements the gap math: alignment, fee netting, `net_best_bps`
and `cap_best_usd` all come from `executable.build_executable_series`, and run
detection from `scanner._find_runs`, so the streaming results are identical to
what a single-frame run would produce apart from day-boundary effects (an
excursion straddling UTC midnight is counted once per side; durations are
sub-second to seconds, so this is immaterial at ~11 boundaries).

Outage masking
--------------
Collector-side feed outages must not be read as a quiet market NOR as an
opportunity. `mask_windows` drops rows inside each [start, end) window before
alignment, and the masked seconds are removed from the active-time denominator
so per-day dollar figures can be reported as rates.

Size units
----------
The two venues quote size in different units (Kraken base currency, OKX
contracts), so this driver resolves the per-venue multipliers from the asset
before calling the analyzer - see `collect_top_of_book.size_multiplier`. Pass
`asset` to `asset_day_records`, or `cap_usd` silently mixes units.

Per-excursion conventions (one row per excursion, evaluated at its BEST moment)
------------------------------------------------------------------------------
Each excursion is a maximal run of `net_best_bps > 0`. Its single representative
row is the run's argmax of `net_best_bps`; `cap_usd`, `net_bps`, `direction` and
`okx_leg_binds` are all read from THAT row, so capacity and edge are internally
consistent (taking max capacity and max edge independently would overstate the
product).

    capturable_usd = cap_usd * net_bps / 10000

Duration follows `executable.analyze_executable`: idx[end_exclusive] - idx[start],
i.e. from the first net-positive update to the update that closed the gap.

Diagnostic only. No orders.

CLI:
    python -m src.strategies.dislocation.streaming --dir data/tob
"""

from __future__ import annotations

import argparse
import glob
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.data.collect_top_of_book import TOB_COLUMNS, size_multiplier
from src.strategies.dislocation.executable import (
    ExecutableConfig,
    _detect_outages,
    build_executable_series,
)
from src.strategies.dislocation.scanner import _find_runs

logger = logging.getLogger(__name__)

QUOTE_COLS = ["bid", "bid_size", "ask", "ask_size"]

# Read dtypes: prices stay float64 (bps math on a ~$84k mid needs the mantissa),
# venue/asset are low-cardinality strings.
_READ_DTYPES = {
    "ts_recv": "int64",
    "ts_exchange": "float64",  # nullable in the collector -> may be blank
    "venue": "category",
    "asset": "category",
    "bid": "float64",
    "bid_size": "float64",
    "ask": "float64",
    "ask_size": "float64",
}


# ============================================================
# Config
# ============================================================

@dataclass
class StreamConfig:
    """Parameters for the chunked multi-day driver.

    Attributes:
        chunksize: rows per `read_csv` chunk. Bounds peak memory of the raw read;
            the retained per-asset slice is far smaller than the chunk.
        mask_windows: [(start, end)] UTC timestamps to exclude from ALL metrics
            (collector/feed outage windows). Half-open: start <= ts < end.
        executable: the underlying analyzer config (fees, staleness tolerance,
            outage gap threshold).
        min_rows: skip an (asset, day) slice with fewer aligned rows than this.
    """

    chunksize: int = 1_000_000
    mask_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)
    executable: ExecutableConfig = field(default_factory=ExecutableConfig)
    min_rows: int = 3


# ============================================================
# Chunked per-(day, asset) loading
# ============================================================

def daily_files(out_dir: Path | str) -> list[Path]:
    """Sorted daily quote CSVs under `out_dir`, excluding the `tob_events_*` logs."""
    out_dir = Path(out_dir)
    return sorted(
        Path(f) for f in glob.glob(str(out_dir / "tob_*.csv"))
        if "events" not in Path(f).name
    )


def mask_seconds(
    start: pd.Timestamp, end: pd.Timestamp,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> float:
    """Total seconds of `windows` overlapping [start, end] (clipped, non-negative)."""
    total = 0.0
    for w0, w1 in windows:
        lo, hi = max(start, w0), min(end, w1)
        if hi > lo:
            total += (hi - lo).total_seconds()
    return total


def load_asset_day(
    path: Path | str, asset: str, config: StreamConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ONE daily CSV in chunks, keeping only `asset`'s rows, split by venue.

    Only the target asset's rows survive each chunk, so peak memory is one chunk
    plus the (much smaller) retained slice -- never the whole file.

    Args:
        path: a single `tob_YYYYMMDD.csv`.
        asset: asset symbol, e.g. "BTC" (case-insensitive).
        config: driver config; supplies `chunksize` and `mask_windows`.

    Returns:
        (kraken_q, okx_q), each indexed by ts_recv (UTC) with columns
        [bid, bid_size, ask, ask_size]. Either may be empty.
    """
    asset = asset.upper()
    keep: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path, usecols=TOB_COLUMNS, dtype=_READ_DTYPES, chunksize=config.chunksize,
    )
    for chunk in reader:
        sl = chunk[chunk["asset"] == asset]
        if not sl.empty:
            keep.append(sl[["ts_recv", "venue"] + QUOTE_COLS])

    cols = pd.Index(QUOTE_COLS)
    if not keep:
        empty = pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz="UTC", name="ts_recv"))
        return empty, empty.copy()

    df = pd.concat(keep, ignore_index=True)
    del keep
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], unit="ms", utc=True)
    df = df.set_index("ts_recv").sort_index()

    for w0, w1 in config.mask_windows:
        df = df[(df.index < w0) | (df.index >= w1)]

    # .copy() so each venue slice owns its block and the parent can be freed.
    k = df[df["venue"] == "kraken"][QUOTE_COLS].copy()
    o = df[df["venue"] == "okx"][QUOTE_COLS].copy()
    return k, o


# ============================================================
# Per-(day, asset) excursion extraction
# ============================================================

def asset_day_records(
    kraken_q: pd.DataFrame, okx_q: pd.DataFrame, config: StreamConfig,
    asset: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-excursion records + coverage stats for one (asset, day) slice.

    Args:
        kraken_q, okx_q: one asset-day's quotes per venue.
        config: driver config.
        asset: the asset these quotes are for. Supplying it resolves the
            per-venue SIZE multipliers (OKX quotes contracts, Kraken base
            units), without which `cap_usd` mixes units. None keeps the
            config's multipliers as-is, for synthetic base-unit input.

    Returns:
        (records, stats). Each record describes one net-positive excursion
        evaluated at its peak-net row (see module docstring). `stats` carries the
        slice's aligned-row count, covered/active seconds, detected outages, and
        `best_net_bps` -- the max `net_best_bps` over the WHOLE slice, which is
        the control metric (meaningful even when no excursion exists).
    """
    empty_stats = {
        "n_aligned": 0, "covered_seconds": 0.0, "active_seconds": 0.0,
        "n_outages": 0, "outage_seconds": 0.0, "best_net_bps": float("nan"),
    }
    if len(kraken_q) < config.min_rows or len(okx_q) < config.min_rows:
        return [], empty_stats

    ec = config.executable
    if asset is not None:
        ec = replace(ec,
                     kraken_size_mult=size_multiplier("kraken", asset),
                     okx_size_mult=size_multiplier("okx", asset))
    m = build_executable_series(kraken_q, okx_q, ec)
    if len(m) < config.min_rows:
        return [], empty_stats

    idx = m.index
    net = m["net_best_bps"].to_numpy(dtype=float)
    cap = m["cap_best_usd"].to_numpy(dtype=float)
    mid = m["mid"].to_numpy(dtype=float)
    k_dir = (m["net_sell_k_buy_o_bps"] >= m["net_sell_o_buy_k_bps"]).to_numpy()
    okx_binds = m["okx_leg_binds"].to_numpy(dtype=bool)
    n = len(net)

    n_outages, outage_seconds = _detect_outages(idx, config.executable.outage_gap_sec)
    covered = (idx[-1] - idx[0]).total_seconds()
    masked = mask_seconds(idx[0], idx[-1], config.mask_windows)
    stats = {
        "n_aligned": n,
        "covered_seconds": covered,
        "active_seconds": max(covered - outage_seconds - masked, 0.0),
        "n_outages": n_outages,
        "outage_seconds": outage_seconds,
        "masked_seconds": masked,
        "best_net_bps": float(net.max()),
        "start": idx[0],
        "end": idx[-1],
    }

    records: list[dict[str, Any]] = []
    for s, e in _find_runs(net > 0):
        end = e if e < n else n - 1
        j = s + int(np.argmax(net[s:e]))          # the excursion's best moment
        cap_j = cap[j]
        net_j = float(net[j])
        records.append({
            "start": idx[s],
            "hour": int(idx[s].hour),
            "weekday": int(idx[s].weekday()),     # 0=Mon .. 6=Sun
            "duration_sec": float((idx[end] - idx[s]).total_seconds()),
            "n_updates": int(e - s),
            "net_bps": net_j,
            "cap_usd": float(cap_j) if np.isfinite(cap_j) else float("nan"),
            "mid": float(mid[j]),
            "capturable_usd": float(cap_j) * net_j / 10000.0 if np.isfinite(cap_j) else float("nan"),
            "direction": "sell_k_buy_o" if k_dir[j] else "sell_o_buy_k",
            "okx_leg_binds": bool(okx_binds[j]),
        })
    return records, stats


def stream_asset(
    out_dir: Path | str, asset: str, config: StreamConfig,
) -> Iterator[tuple[str, list[dict[str, Any]], dict[str, Any]]]:
    """Yield (day_str, records, stats) for one asset, one daily file at a time."""
    for path in daily_files(out_dir):
        day = path.stem.replace("tob_", "")
        k, o = load_asset_day(path, asset, config)
        recs, stats = asset_day_records(k, o, config, asset)
        del k, o
        logger.info("%s %s: %d aligned, %d excursions", asset, day, stats["n_aligned"], len(recs))
        yield day, recs, stats


def analyze_streaming(
    out_dir: Path | str, asset: str, config: StreamConfig,
) -> dict[str, Any]:
    """Full-collection executable-gap result for one asset, built day by day.

    Returns a dict with `per_day` (one entry per daily file) and `records` (one
    entry per excursion across the whole collection). Aggregation/reporting is
    left to the caller so partial days can be rate-normalised.
    """
    per_day: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for day, recs, stats in stream_asset(out_dir, asset, config):
        if stats["n_aligned"] == 0:
            continue
        for r in recs:
            r["day"] = day
        records.extend(recs)
        per_day[day] = {
            **stats,
            "n_excursions": len(recs),
            "capturable_usd": float(np.nansum([r["capturable_usd"] for r in recs])) if recs else 0.0,
        }
    return {"asset": asset, "per_day": per_day, "records": records}


# ============================================================
# CLI
# ============================================================

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Chunked multi-day driver for the executable-gap analyzer (Phase B).")
    parser.add_argument("--dir", default="data/tob")
    parser.add_argument("--assets", default="ARB,NEAR,UNI,BTC")
    parser.add_argument("--fee-bps-per-side", type=float, default=5.0)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--mask", action="append", default=[],
                        help="UTC outage window to exclude, 'START/END' ISO8601. Repeatable.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    windows = []
    for w in args.mask:
        a, b = w.split("/")
        windows.append((pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")))

    config = StreamConfig(
        chunksize=args.chunksize,
        mask_windows=windows,
        executable=ExecutableConfig(fee_bps_per_side=args.fee_bps_per_side,
                                    kraken_fee_bps=args.fee_bps_per_side,
                                    okx_fee_bps=args.fee_bps_per_side),
    )

    for asset in [a.strip().upper() for a in args.assets.split(",")]:
        res = analyze_streaming(args.dir, asset, config)
        recs, per_day = res["records"], res["per_day"]
        active_h = sum(d["active_seconds"] for d in per_day.values()) / 3600.0
        dollars = float(np.nansum([r["capturable_usd"] for r in recs])) if recs else 0.0
        best = max((d["best_net_bps"] for d in per_day.values()), default=float("nan"))
        print(f"{asset:>5}: {len(recs):>7,} excursions over {active_h:7.2f} active h "
              f"| best net {best:7.2f} bps | capturable ${dollars:,.0f}")


if __name__ == "__main__":
    main()
