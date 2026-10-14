"""
src/strategies/dislocation/executable.py - Executable-gap analysis (Phase B).

Phase A measured candle-close gaps; those cannot be traded (stale last-trade
prints, spreads the candles can't see, sub-sampling-interval reversion). This
module consumes the Phase B top-of-book collection and computes the EXECUTABLE
gap: could you actually cross both books, net of taker fees, and profit?

Two directions, both crossing the spread on each venue (taker on both legs):
    sell_kraken_buy_okx = (kraken_bid - okx_ask) / mid
    sell_okx_buy_kraken = (okx_bid - kraken_ask) / mid
Net of fees: subtract taker fee on BOTH legs. `net_best` is the better direction.
An "executable excursion" is a run where net_best > 0. For each we also report the
SIZE-CAPPED dollar capacity = min(bid_size, ask_size) x mid on the winning
direction - the "is the edge more than retail-dust?" question.

Alignment uses ONE clock (collector ts_recv) via an as-of join with a tight
staleness tolerance: a venue quote older than the tolerance is excluded, never
carried forward (a stale quote is exactly what produces phantom candle gaps).

Feed outages are detected from time gaps in the merged series and reported
separately: a dead feed must not be read as a quiet (no-opportunity) market.

Diagnostic only. No orders.

CLI:
    python -m src.strategies.dislocation.executable --dir data/tob
"""

from __future__ import annotations

import argparse
import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.collect_top_of_book import (
    KRAKEN_TAKER_BPS,
    OKX_TAKER_BPS,
    size_multiplier,
)
from src.strategies.dislocation.scanner import _find_runs, _half_life

logger = logging.getLogger(__name__)


# ============================================================
# Config
# ============================================================

@dataclass
class ExecutableConfig:
    """Parameters for executable-gap analysis.

    Attributes:
        staleness_tolerance: max age of a venue quote in the as-of join (pandas
            offset). Older quotes are excluded, never forward-filled.
        fee_bps_per_side: taker fee per leg in bps. Both legs are taker, so total
            deducted = 2 x this. Defaults to the collector's published-taker
            constants (Kraken 5 bps, OKX 5 bps); override per venue via the two
            explicit fields if they diverge.
        kraken_fee_bps, okx_fee_bps: per-venue taker overrides (default to the
            published 5 bps each).
        outage_gap_sec: a gap longer than this between consecutive merged updates
            is flagged as a feed outage and excluded from the active-time base.
        kraken_size_mult, okx_size_mult: base units per quoted size unit, per
            venue. `cap_best_usd` takes a min() ACROSS the two venues' sizes, so
            both must be in base units first - OKX quotes SWAP size in contracts
            (ctVal), Kraken in base currency. Defaults are 1.0/1.0, which is
            correct only for already-converted or synthetic input; use
            `ExecutableConfig.for_asset()` to resolve them from the real ctVal
            table. Leaving OKX at 1.0 misstates capacity by the contract value
            (100x on BTC).
    """

    staleness_tolerance: str = "500ms"
    fee_bps_per_side: float = 5.0
    kraken_fee_bps: float = KRAKEN_TAKER_BPS
    okx_fee_bps: float = OKX_TAKER_BPS
    outage_gap_sec: float = 10.0
    kraken_size_mult: float = 1.0
    okx_size_mult: float = 1.0

    @classmethod
    def for_asset(cls, asset: str, **kwargs: Any) -> "ExecutableConfig":
        """Config with both venues' size multipliers resolved for `asset`.

        Prefer this over the bare constructor whenever the input is real
        collector data, so `cap_best_usd` compares like with like.
        """
        kwargs.setdefault("kraken_size_mult", size_multiplier("kraken", asset))
        kwargs.setdefault("okx_size_mult", size_multiplier("okx", asset))
        return cls(**kwargs)


# ============================================================
# Loading collected quotes
# ============================================================

def load_tob(out_dir: Path | str, asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one asset's collected quotes, split by venue, indexed by ts_recv (UTC).

    Reads every `tob_*.csv` (not the `tob_events_*` files) under `out_dir`.
    Returns (kraken_q, okx_q), each with columns [bid, bid_size, ask, ask_size].
    """
    out_dir = Path(out_dir)
    files = [f for f in glob.glob(str(out_dir / "tob_*.csv")) if "events" not in Path(f).name]
    if not files:
        raise FileNotFoundError(f"No tob_*.csv files under {out_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["asset"] == asset.upper()]
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], unit="ms", utc=True)
    df = df.set_index("ts_recv").sort_index()
    cols = ["bid", "bid_size", "ask", "ask_size"]
    k = df[df["venue"] == "kraken"][cols]
    o = df[df["venue"] == "okx"][cols]
    return k, o


# ============================================================
# As-of alignment + executable gaps
# ============================================================

def build_executable_series(
    kraken_q: pd.DataFrame, okx_q: pd.DataFrame, config: ExecutableConfig
) -> pd.DataFrame:
    """Align both venues' quotes on the collector clock and compute net gaps.

    At every update time (union of both venues' ts_recv), take each venue's most
    recent quote within `staleness_tolerance` (backward as-of join); rows where
    either venue is stale are dropped. Returns a DataFrame with the two gross and
    net (fee-deducted) gaps in bps, `net_best_bps`, `cap_best_usd`, and
    `okx_leg_binds` (whether OKX's depth was the binding side).

    `cap_best_usd` converts both venues' sizes to base units via the config's
    size multipliers BEFORE taking the min(); see `ExecutableConfig.for_asset`.
    """
    for name, q in (("kraken_q", kraken_q), ("okx_q", okx_q)):
        if not isinstance(q.index, pd.DatetimeIndex):
            raise ValueError(f"{name} must be indexed by a DatetimeIndex (ts_recv)")
        for c in ("bid", "ask", "bid_size", "ask_size"):
            if c not in q.columns:
                raise ValueError(f"{name} missing column {c!r}")

    tol = pd.Timedelta(config.staleness_tolerance)
    k = kraken_q.add_prefix("k_").sort_index()
    o = okx_q.add_prefix("o_").sort_index()
    allts = k.index.union(o.index).sort_values()
    base = pd.DataFrame(index=allts)

    # Backward as-of within tolerance == "latest quote, if not stale".
    kk = pd.merge_asof(base, k, left_index=True, right_index=True,
                       direction="backward", tolerance=tol)
    oo = pd.merge_asof(base, o, left_index=True, right_index=True,
                       direction="backward", tolerance=tol)
    m = pd.concat([kk, oo], axis=1).dropna()
    if m.empty:
        raise ValueError("No overlapping non-stale quotes between the two venues.")

    mid = (m["k_bid"] + m["k_ask"] + m["o_bid"] + m["o_ask"]) / 4.0
    total_fee = config.kraken_fee_bps + config.okx_fee_bps  # taker on both legs

    m = m.assign(mid=mid)
    m["gross_sell_k_buy_o_bps"] = 10000.0 * (m["k_bid"] - m["o_ask"]) / mid
    m["gross_sell_o_buy_k_bps"] = 10000.0 * (m["o_bid"] - m["k_ask"]) / mid
    m["net_sell_k_buy_o_bps"] = m["gross_sell_k_buy_o_bps"] - total_fee
    m["net_sell_o_buy_k_bps"] = m["gross_sell_o_buy_k_bps"] - total_fee
    m["net_best_bps"] = m[["net_sell_k_buy_o_bps", "net_sell_o_buy_k_bps"]].max(axis=1)

    # Size-capped USD capacity of the WINNING direction at each row.
    # CONVERT TO BASE UNITS BEFORE THE min(): Kraken quotes base currency, OKX
    # quotes contracts. Taking the min() of raw sizes compares different units.
    k_bid_base = m["k_bid_size"] * config.kraken_size_mult
    k_ask_base = m["k_ask_size"] * config.kraken_size_mult
    o_bid_base = m["o_bid_size"] * config.okx_size_mult
    o_ask_base = m["o_ask_size"] * config.okx_size_mult
    cap_a = np.minimum(k_bid_base, o_ask_base) * mid  # sell kraken / buy okx
    cap_b = np.minimum(o_bid_base, k_ask_base) * mid  # sell okx / buy kraken
    # Which venue's depth bound the trade - the diagnostic that reveals a units
    # error (a leg that binds ~always is a red flag).
    m["okx_leg_binds"] = np.where(
        m["net_sell_k_buy_o_bps"] >= m["net_sell_o_buy_k_bps"],
        o_ask_base <= k_bid_base, o_bid_base <= k_ask_base)
    m["cap_best_usd"] = np.where(
        m["net_sell_k_buy_o_bps"] >= m["net_sell_o_buy_k_bps"], cap_a, cap_b)
    return m


# ============================================================
# Analysis
# ============================================================

def _detect_outages(index: pd.DatetimeIndex, outage_gap_sec: float) -> tuple[int, float]:
    """Count time gaps > outage_gap_sec in the merged series and total their seconds."""
    if len(index) < 2:
        return 0, 0.0
    dt = index.to_series().diff().dt.total_seconds().to_numpy()[1:]
    mask = dt > outage_gap_sec
    return int(mask.sum()), float(dt[mask].sum())


def analyze_executable(
    kraken_q: pd.DataFrame, okx_q: pd.DataFrame, config: ExecutableConfig
) -> dict[str, Any]:
    """Executable-gap diagnostics for one asset. See module docstring."""
    m = build_executable_series(kraken_q, okx_q, config)
    if len(m) < 3:
        raise ValueError("Fewer than 3 aligned quotes; nothing to analyze")

    idx = m.index
    net = m["net_best_bps"].to_numpy(dtype=float)
    cap = m["cap_best_usd"].to_numpy(dtype=float)
    n = len(net)

    # Feed outages: flagged and removed from the active-time denominator, so a dead
    # feed is never counted as a quiet market.
    n_outages, outage_seconds = _detect_outages(idx, config.outage_gap_sec)
    covered_seconds = (idx[-1] - idx[0]).total_seconds()
    active_seconds = max(covered_seconds - outage_seconds, 1e-9)
    active_days = active_seconds / 86400.0

    # Executable excursions: runs where net_best > 0 (a real, fee-cleared edge).
    mask = net > 0
    runs = _find_runs(mask)
    durations_sec, peaks_bps, caps_usd = [], [], []
    for s, e in runs:
        end = e if e < n else n - 1
        durations_sec.append((idx[end] - idx[s]).total_seconds())
        peaks_bps.append(float(net[s:e].max()))
        caps_usd.append(float(np.nanmax(cap[s:e])) if np.any(np.isfinite(cap[s:e])) else float("nan"))
    durations_sec = np.array(durations_sec, dtype=float)
    peaks_bps = np.array(peaks_bps, dtype=float)
    caps_usd = np.array(caps_usd, dtype=float)

    phi, hl_updates = _half_life(net)  # per-update (irregular sampling), rough persistence

    def _mean(a):
        return float(np.nanmean(a)) if len(a) else float("nan")

    def _median(a):
        return float(np.nanmedian(a)) if len(a) else float("nan")

    return {
        "coverage": {
            "start": idx[0], "end": idx[-1],
            "n_aligned_updates": int(n),
            "covered_seconds": covered_seconds,
            "active_seconds": active_seconds,
        },
        "outages": {"n_outages": n_outages, "outage_seconds": outage_seconds},
        "executable": {
            "n_excursions": len(runs),
            "excursions_per_day": len(runs) / active_days,
            "frac_time_net_positive": float(mask.mean()),
            "mean_duration_sec": _mean(durations_sec),
            "median_duration_sec": _median(durations_sec),
            "mean_peak_net_bps": _mean(peaks_bps),
            "max_net_bps": float(net.max()),
            "mean_capacity_usd": _mean(caps_usd),
            "median_capacity_usd": _median(caps_usd),
        },
        "net_gap_bps": {
            "mean": float(net.mean()), "median": float(np.median(net)),
            "p95": float(np.percentile(net, 95)), "max": float(net.max()),
        },
        "persistence": {"lag1_autocorr_per_update": phi, "half_life_updates": hl_updates},
        "fees": {"kraken_bps": config.kraken_fee_bps, "okx_bps": config.okx_fee_bps,
                 "total_round_trip_bps": config.kraken_fee_bps + config.okx_fee_bps},
    }


def compare_to_candle(executable_per_day: float, candle_per_day: float) -> dict[str, float]:
    """Ratio of executable to Phase A candle excursions/day (expected << 1)."""
    ratio = executable_per_day / candle_per_day if candle_per_day > 0 else float("nan")
    return {"executable_per_day": executable_per_day,
            "candle_per_day": candle_per_day, "ratio": ratio}


# ============================================================
# Pretty printing
# ============================================================

def format_executable(asset: str, a: dict[str, Any]) -> str:
    cov, ex, out = a["coverage"], a["executable"], a["outages"]
    lines = ["-" * 74]
    lines.append(f"  {asset}  |  {cov['start']} -> {cov['end']}  "
                 f"({cov['n_aligned_updates']:,} aligned updates, "
                 f"{cov['active_seconds']/3600:.2f}h active)")
    lines.append("-" * 74)
    lines.append(f"    fees (round trip, both legs taker): {a['fees']['total_round_trip_bps']:.1f} bps")
    lines.append(f"    feed outages: {out['n_outages']} (total {out['outage_seconds']:.0f}s, excluded)")
    lines.append(f"    EXECUTABLE excursions (net > 0): {ex['n_excursions']}  "
                 f"= {ex['excursions_per_day']:.2f}/day  "
                 f"(net>0 {ex['frac_time_net_positive']*100:.2f}% of time)")
    if ex["n_excursions"] > 0:
        lines.append(f"      mean dur {ex['mean_duration_sec']:.2f}s (median {ex['median_duration_sec']:.2f}s), "
                     f"mean peak net {ex['mean_peak_net_bps']:.2f} bps, max {ex['max_net_bps']:.2f} bps")
        lines.append(f"      size-capped capacity: mean ${ex['mean_capacity_usd']:,.0f}, "
                     f"median ${ex['median_capacity_usd']:,.0f}")
    else:
        lines.append(f"      (no fee-cleared edge; best net gap {a['net_gap_bps']['max']:.2f} bps)")
    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Executable cross-venue gap analyzer (Phase B).")
    parser.add_argument("--dir", default="data/tob", help="Directory of collected tob_*.csv.")
    parser.add_argument("--assets", default=None,
                        help="Comma-separated assets (default: all found).")
    parser.add_argument("--fee-bps-per-side", type=float, default=5.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = ExecutableConfig(fee_bps_per_side=args.fee_bps_per_side,
                              kraken_fee_bps=args.fee_bps_per_side,
                              okx_fee_bps=args.fee_bps_per_side)

    # Discover assets present in the collection.
    files = [f for f in glob.glob(str(Path(args.dir) / "tob_*.csv")) if "events" not in Path(f).name]
    if not files:
        print(f"No tob_*.csv under {args.dir}")
        return
    present = sorted(pd.concat([pd.read_csv(f, usecols=["asset"]) for f in files])["asset"].unique())
    assets = [a.strip().upper() for a in args.assets.split(",")] if args.assets else present

    print("=" * 74)
    print("  EXECUTABLE CROSS-VENUE GAP ANALYSIS (Phase B, top-of-book, net of fees)")
    print("  BTC is the control - expect ~0 executable excursions.")
    print("=" * 74)
    for asset in assets:
        try:
            k, o = load_tob(args.dir, asset)
            a = analyze_executable(k, o, config)
            print(format_executable(asset, a))
        except Exception as e:
            print(f"  {asset}: skipped ({e})")
    print("=" * 74)
    print("  Diagnostic only. No orders. Believe a candidate only if executable excursions survive.")


if __name__ == "__main__":
    main()
