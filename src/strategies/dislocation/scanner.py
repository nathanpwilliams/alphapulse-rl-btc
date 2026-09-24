"""
src/strategies/dislocation/scanner.py - Cross-venue price dislocation scanner.

Phase A diagnostic for a STRUCTURAL edge: measure the price gap between the same
asset on Kraken (USD-quoted PF_ perp) and OKX (USDT-quoted swap) at 5-minute
resolution, across a multi-asset basket, and characterize whether gaps are large
enough to clear costs AND transient enough to be arbitrageable. Diagnostic only -
it computes and prints; it draws no viability verdict.

BTC and ETH are efficient-market CONTROLS: the scan is only trustworthy if they
show essentially no cost-clearing transient excursions.

Two flavors of the gap matter, and the analyzer reports BOTH:
  * RAW gap: kraken vs okx directly. Includes the persistent USD/USDT basis +
    any venue premium - a large, sign-stable mean here is STRUCTURAL (a
    capital-transfer friction), not an opportunity.
  * MEAN-REMOVED (demeaned) gap: raw gap minus its mean. This strips the
    structural premium and isolates the TRANSIENT dislocation - the part a
    market-neutral arb could actually harvest. The demeaned excursions are the
    headline arbitrage signal.

Sign convention: gap_bps = 10000 * (kraken_close - okx_close) / okx_close.
Positive => Kraken richer than OKX.

CLI:
    python -m src.strategies.dislocation.scanner
    python -m src.strategies.dislocation.scanner --cost-bps 20 --days 30
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

# Bar length in minutes per resolution, for duration and per-day conversions.
RESOLUTION_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
                      "1h": 60, "4h": 240, "12h": 720, "1d": 1440}
MINUTES_PER_DAY = 1440


# ============================================================
# Config
# ============================================================

@dataclass
class DislocationConfig:
    """Parameters for the cross-venue dislocation scan.

    Attributes:
        resolution: bar size (must be a key of RESOLUTION_MINUTES).
        round_trip_cost_bps: the default cost hurdle in bps (both venues' fees +
            assumed spread/slippage). A dislocation must exceed this to be
            actionable. The scanner sorts on excursions/day at this cost.
        cost_grid_bps: cost levels for the sensitivity sweep.
        align_tolerance: max timestamp mismatch when aligning the two venues
            (pandas offset string). Bars without a match within tolerance are
            DROPPED, never forward-filled (which would fabricate a stale gap).
        reversion_fraction: an excursion is considered "reverted" once |gap| falls
            below this fraction of the hurdle (0.5 => half the hurdle).
    """

    resolution: str = "5m"
    round_trip_cost_bps: float = 20.0
    cost_grid_bps: tuple = (10, 20, 30)
    align_tolerance: str = "1min"
    reversion_fraction: float = 0.5


# ============================================================
# Alignment
# ============================================================

def align_venues(
    kraken_df: pd.DataFrame, okx_df: pd.DataFrame, config: DislocationConfig
) -> pd.DataFrame:
    """Align the two venues' closes on a common grid and compute the per-bar gap.

    Restricts to the overlapping time window, then `merge_asof` with
    direction="nearest" and a tight tolerance so each Kraken bar is matched to the
    OKX bar at (essentially) the same instant. Unmatched bars are dropped - NO
    forward-fill, so a data gap on one venue never fabricates a stale gap.

    Returns:
        DataFrame indexed by the aligned timestamps with columns
        [kraken_close, okx_close, gap_bps].
    """
    for name, df in (("kraken_df", kraken_df), ("okx_df", okx_df)):
        if "close" not in df.columns:
            raise ValueError(f"{name} must have a 'close' column")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{name} must have a DatetimeIndex")

    k = kraken_df["close"].rename("kraken_close").sort_index()
    o = okx_df["close"].rename("okx_close").sort_index()

    # Disjoint ranges => nothing to compare. (Sub-bar jitter is fine; merge_asof's
    # tolerance handles boundary bars, so we do NOT pre-trim to first/last, which
    # would wrongly drop jittered edge bars.)
    if k.index[-1] < o.index[0] or o.index[-1] < k.index[0]:
        raise ValueError(
            f"Kraken and OKX candles do not overlap: "
            f"Kraken [{k.index[0]}..{k.index[-1]}], OKX [{o.index[0]}..{o.index[-1]}]."
        )

    tol = pd.Timedelta(config.align_tolerance)
    # nearest-within-tolerance match; bars with no counterpart in range are dropped
    # (never forward-filled), so the result covers only the true overlap.
    merged = pd.merge_asof(
        k.to_frame(), o.to_frame(),
        left_index=True, right_index=True,
        direction="nearest", tolerance=tol,
    )
    merged = merged.dropna()
    if merged.empty:
        raise ValueError("Kraken and OKX candles do not overlap within tolerance.")
    # gap in bps: positive => Kraken richer than OKX. (kraken - okx) / okx * 1e4.
    merged["gap_bps"] = 10000.0 * (merged["kraken_close"] - merged["okx_close"]) / merged["okx_close"]
    return merged


# ============================================================
# Excursion analysis
# ============================================================

def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) index pairs for each maximal run of True."""
    runs: list[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _excursions(
    abs_gap: np.ndarray, threshold: float, revert_level: float, bar_minutes: float
) -> dict[str, Any]:
    """Excursion statistics for |gap| exceeding `threshold`.

    An excursion is a run of consecutive bars with abs_gap > threshold. Time-to-
    revert is measured from each excursion's START forward to the first bar where
    abs_gap < `revert_level` (a stricter drop than merely exiting the excursion).
    """
    n = len(abs_gap)
    mask = abs_gap > threshold
    runs = _find_runs(mask)
    n_days = n * bar_minutes / MINUTES_PER_DAY

    if not runs:
        return {
            "n_excursions": 0, "excursions_per_day": 0.0,
            "frac_bars_in_excursion": 0.0,
            "mean_duration_bars": float("nan"), "median_duration_bars": float("nan"),
            "mean_duration_min": float("nan"),
            "mean_time_to_revert_bars": float("nan"), "mean_time_to_revert_min": float("nan"),
            "mean_peak_abs_gap_bps": float("nan"),
        }

    durations = np.array([e - s for s, e in runs], dtype=float)
    peaks = np.array([abs_gap[s:e].max() for s, e in runs], dtype=float)

    revert_bars = []
    for s, _e in runs:
        j = s
        while j < n and abs_gap[j] >= revert_level:
            j += 1
        if j < n:  # reverted within the observed window
            revert_bars.append(j - s)
    revert_bars = np.array(revert_bars, dtype=float)

    return {
        "n_excursions": len(runs),
        "excursions_per_day": len(runs) / n_days if n_days > 0 else 0.0,
        "frac_bars_in_excursion": float(mask.sum() / n),
        "mean_duration_bars": float(durations.mean()),
        "median_duration_bars": float(np.median(durations)),
        "mean_duration_min": float(durations.mean() * bar_minutes),
        "mean_time_to_revert_bars": float(revert_bars.mean()) if len(revert_bars) else float("nan"),
        "mean_time_to_revert_min": float(revert_bars.mean() * bar_minutes) if len(revert_bars) else float("nan"),
        "mean_peak_abs_gap_bps": float(peaks.mean()),
    }


# ============================================================
# Per-asset analysis
# ============================================================

def _half_life(gap: np.ndarray) -> tuple[float, float]:
    """Lag-1 autocorrelation and AR(1) mean-reversion half-life (in bars).

    half_life = -ln(2)/ln(phi) for 0<phi<1. phi<=0 => ~0 bars (instant revert);
    phi>=1 => inf (no reversion).
    """
    # A constant (or near-constant) series has undefined autocorrelation; guard the
    # lagged sub-arrays too so np.corrcoef never divides by a zero std.
    if len(gap) < 3 or np.std(gap[:-1]) < 1e-12 or np.std(gap[1:]) < 1e-12:
        return float("nan"), float("nan")
    phi = float(np.corrcoef(gap[:-1], gap[1:])[0, 1])
    if phi <= 0:
        hl = 0.0
    elif phi >= 1:
        hl = float("inf")
    else:
        hl = -np.log(2) / np.log(phi)
    return phi, hl


def analyze_dislocation(
    kraken_df: pd.DataFrame, okx_df: pd.DataFrame, config: DislocationConfig
) -> dict[str, Any]:
    """Full per-asset dislocation diagnostics. See module docstring for the two
    gap flavors (raw vs mean-removed) and the sign convention."""
    bar_minutes = RESOLUTION_MINUTES[config.resolution]
    aligned = align_venues(kraken_df, okx_df, config)
    if len(aligned) < 3:
        raise ValueError("Aligned overlap has fewer than 3 bars; nothing to analyze")

    gap = aligned["gap_bps"].to_numpy(dtype=float)
    abs_gap = np.abs(gap)
    n = len(gap)
    mean_gap = float(gap.mean())

    # ---- 1. Coverage ------------------------------------------------------
    coverage = {
        "overlap_start": aligned.index[0],
        "overlap_end": aligned.index[-1],
        "n_aligned_bars": int(n),
        "days": n * bar_minutes / MINUTES_PER_DAY,
    }

    # ---- 2. Gap magnitude -------------------------------------------------
    magnitude = {
        "mean_bps": mean_gap,
        "median_bps": float(np.median(gap)),
        "std_bps": float(gap.std()),
        "min_bps": float(gap.min()),
        "max_bps": float(gap.max()),
        "mean_abs_bps": float(abs_gap.mean()),
        "p50_abs_bps": float(np.percentile(abs_gap, 50)),
        "p90_abs_bps": float(np.percentile(abs_gap, 90)),
        "p95_abs_bps": float(np.percentile(abs_gap, 95)),
        "p99_abs_bps": float(np.percentile(abs_gap, 99)),
    }

    # ---- 3. Structural premium -------------------------------------------
    # A large, sign-stable mean gap is a persistent premium (likely the USD/USDT
    # basis + venue friction), reported SEPARATELY from transient dislocation.
    signs = np.sign(gap)
    if mean_gap != 0:
        sign_stability = float(np.mean(signs == np.sign(mean_gap)))
    else:
        sign_stability = float(max((gap > 0).mean(), (gap < 0).mean()))
    n_flips = int(np.sum(signs[1:] != signs[:-1])) if n >= 2 else 0
    structural = {
        "mean_gap_bps": mean_gap,
        "sign_stability": sign_stability,
        "frac_positive": float((gap > 0).mean()),
        "sign_flip_fraction": float(n_flips / (n - 1)) if n >= 2 else 0.0,
    }

    # ---- 4. Excursions (raw AND mean-removed), per cost level -------------
    demeaned_abs = np.abs(gap - mean_gap)
    excursions: dict[str, dict[int, dict[str, Any]]] = {"raw": {}, "demeaned": {}}
    for cost in config.cost_grid_bps:
        revert = config.reversion_fraction * cost
        excursions["raw"][int(cost)] = _excursions(abs_gap, cost, revert, bar_minutes)
        excursions["demeaned"][int(cost)] = _excursions(demeaned_abs, cost, revert, bar_minutes)

    # ---- 5. Mean-reversion strength --------------------------------------
    phi, hl_bars = _half_life(gap)
    mean_reversion = {
        "lag1_autocorr": phi,
        "half_life_bars": hl_bars,
        "half_life_min": hl_bars * bar_minutes if np.isfinite(hl_bars) else hl_bars,
    }

    return {
        "coverage": coverage,
        "magnitude": magnitude,
        "structural": structural,
        "excursions": excursions,
        "mean_reversion": mean_reversion,
        "config": {"resolution": config.resolution,
                   "round_trip_cost_bps": config.round_trip_cost_bps,
                   "cost_grid_bps": list(config.cost_grid_bps)},
    }


# ============================================================
# Cross-asset scanner
# ============================================================

def scan_basket(
    basket_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    config: DislocationConfig,
    liquidity: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Run `analyze_dislocation` per asset and return one headline row each.

    Sorted by TRANSIENT (mean-removed) excursions/day at the default cost hurdle -
    the arbitrage-relevant figure. BTC/ETH rows are controls (expected ~0).

    Args:
        basket_data: {asset: (kraken_df, okx_df)} from `load_cross_venue_basket`.
        config: scan parameters.
        liquidity: optional {asset: 24h USD notional} for context.

    Returns:
        DataFrame (one row per successfully-analyzed asset), sorted descending on
        `transient_exc_per_day@{default}`.
    """
    liquidity = liquidity or {}
    default = int(config.round_trip_cost_bps)
    rows: list[dict[str, Any]] = []
    for asset, (kdf, odf) in basket_data.items():
        try:
            a = analyze_dislocation(kdf, odf, config)
        except Exception as e:
            logger.warning("Analysis failed for %s: %s", asset, e)
            continue
        row: dict[str, Any] = {
            "asset": asset,
            "liquidity_usd_24h": liquidity.get(asset, float("nan")),
            "n_bars": a["coverage"]["n_aligned_bars"],
            "mean_abs_gap_bps": a["magnitude"]["mean_abs_bps"],
            "p95_abs_gap_bps": a["magnitude"]["p95_abs_bps"],
            "struct_mean_gap_bps": a["structural"]["mean_gap_bps"],
            "sign_stability": a["structural"]["sign_stability"],
            "lag1_autocorr": a["mean_reversion"]["lag1_autocorr"],
            "half_life_min": a["mean_reversion"]["half_life_min"],
        }
        # Transient (mean-removed) excursions/day at every cost level.
        for cost in config.cost_grid_bps:
            row[f"transient_exc_per_day@{int(cost)}"] = a["excursions"]["demeaned"][int(cost)]["excursions_per_day"]
        # Detail at the default cost (transient).
        d = a["excursions"]["demeaned"][default]
        row["transient_mean_exc_dur_min"] = d["mean_duration_min"]
        row["transient_mean_revert_min"] = d["mean_time_to_revert_min"]
        row["raw_exc_per_day@default"] = a["excursions"]["raw"][default]["excursions_per_day"]
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(f"transient_exc_per_day@{default}", ascending=False).reset_index(drop=True)
    return df


# ============================================================
# Pretty printing
# ============================================================

def format_scan_table(df: pd.DataFrame, config: DislocationConfig) -> str:
    """Render the cross-asset summary table."""
    default = int(config.round_trip_cost_bps)
    lines: list[str] = []
    bar = "=" * 108
    lines.append(bar)
    lines.append(f"  CROSS-VENUE DISLOCATION SCAN (Kraken USD vs OKX USDT, {config.resolution}) "
                 f"- sorted by transient excursions/day @ {default}bps")
    lines.append(bar)
    hdr = (f"  {'asset':<6}{'liq_$24h':>12}{'bars':>7}{'mean|gap|':>10}{'p95|gap|':>10}"
           f"{'struct':>9}{'signStab':>9}{'ac1':>7}{'hl_min':>8}")
    for cost in config.cost_grid_bps:
        hdr += f"{'exc/d@' + str(int(cost)):>10}"
    lines.append(hdr)
    lines.append("  " + "-" * 104)
    for _, r in df.iterrows():
        liq = r["liquidity_usd_24h"]
        liq_s = f"{liq/1e6:,.0f}M" if np.isfinite(liq) and liq > 0 else "n/a"
        hl = r["half_life_min"]
        hl_s = "inf" if hl == float("inf") else (f"{hl:.1f}" if np.isfinite(hl) else "n/a")
        line = (f"  {r['asset']:<6}{liq_s:>12}{int(r['n_bars']):>7}{r['mean_abs_gap_bps']:>10.2f}"
                f"{r['p95_abs_gap_bps']:>10.2f}{r['struct_mean_gap_bps']:>9.2f}"
                f"{r['sign_stability']:>9.2f}{r['lag1_autocorr']:>7.2f}{hl_s:>8}")
        for cost in config.cost_grid_bps:
            line += f"{r[f'transient_exc_per_day@{int(cost)}']:>10.2f}"
        lines.append(line)
    lines.append(bar)
    lines.append("  struct = mean signed gap (bps; persistent premium incl. USD/USDT basis).")
    lines.append("  exc/d@X = TRANSIENT (mean-removed) excursions per day over an X bps hurdle.")
    lines.append("  Controls (BTC/ETH) should show ~0 transient excursions if the scan is sound.")
    lines.append("  Diagnostic only. No viability verdict is drawn here.")
    lines.append(bar)
    return "\n".join(lines)


def format_asset_detail(asset: str, a: dict[str, Any], config: DislocationConfig) -> str:
    """Render the per-asset detail block."""
    cov, mag, st, mr = a["coverage"], a["magnitude"], a["structural"], a["mean_reversion"]
    default = int(config.round_trip_cost_bps)
    lines: list[str] = []
    lines.append("-" * 72)
    lines.append(f"  {asset}  |  {cov['overlap_start']} -> {cov['overlap_end']}  "
                 f"({cov['n_aligned_bars']:,} bars, {cov['days']:.1f}d)")
    lines.append("-" * 72)
    lines.append(f"    gap bps: mean {mag['mean_bps']:+.2f}  median {mag['median_bps']:+.2f}  "
                 f"std {mag['std_bps']:.2f}  min {mag['min_bps']:+.2f}  max {mag['max_bps']:+.2f}")
    lines.append(f"    |gap| bps: mean {mag['mean_abs_bps']:.2f}  "
                 f"p50 {mag['p50_abs_bps']:.2f}  p90 {mag['p90_abs_bps']:.2f}  "
                 f"p95 {mag['p95_abs_bps']:.2f}  p99 {mag['p99_abs_bps']:.2f}")
    lines.append(f"    structural: mean gap {st['mean_gap_bps']:+.2f} bps, sign-stability "
                 f"{st['sign_stability']*100:.1f}%, flip-frac {st['sign_flip_fraction']*100:.1f}%")
    lines.append(f"    mean-reversion: lag1 autocorr {mr['lag1_autocorr']:.3f}, "
                 f"half-life {mr['half_life_min'] if mr['half_life_min']!=float('inf') else 'inf'} min")
    lines.append(f"    excursions over hurdle (raw | transient/mean-removed):")
    lines.append(f"      {'cost':>5}  {'raw/day':>8} {'trans/day':>9} {'trans_dur_min':>13} {'trans_revert_min':>16} {'trans_peak_bps':>14}")
    for cost in config.cost_grid_bps:
        raw = a["excursions"]["raw"][int(cost)]
        dm = a["excursions"]["demeaned"][int(cost)]
        def fmt(x):
            return "  -  " if (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"
        lines.append(f"      {int(cost):>5}  {raw['excursions_per_day']:>8.2f} "
                     f"{dm['excursions_per_day']:>9.2f} {fmt(dm['mean_duration_min']):>13} "
                     f"{fmt(dm['mean_time_to_revert_min']):>16} {fmt(dm['mean_peak_abs_gap_bps']):>14}")
    return "\n".join(lines)


# ============================================================
# CLI entrypoint
# ============================================================

def main(argv: list[str] | None = None) -> None:
    """Load the basket from both venues live, scan, and print the report."""
    from src.data.load_cross_venue import (
        DEFAULT_BASKET, fetch_basket_liquidity, load_cross_venue_basket,
    )

    parser = argparse.ArgumentParser(
        description="Cross-venue price dislocation scanner (Phase A diagnostic)."
    )
    parser.add_argument("--cost-bps", type=float, default=20.0,
                        help="Default cost hurdle in bps (sort key).")
    parser.add_argument("--resolution", default="5m", help="Bar size (default 5m).")
    parser.add_argument("--days", type=int, default=30,
                        help="Overlap window length in days (default 30).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = DislocationConfig(resolution=args.resolution, round_trip_cost_bps=args.cost_bps)
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=args.days)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")

    logger.info("Loading basket from both venues (%s, %s -> %s)...", config.resolution, start, end)
    basket_data = load_cross_venue_basket(DEFAULT_BASKET, resolution=config.resolution,
                                          start=start, end=end)
    liquidity = fetch_basket_liquidity(DEFAULT_BASKET)

    table = scan_basket(basket_data, config, liquidity)
    print(format_scan_table(table, config))

    # Per-asset detail: top 3 by transient opportunity + BTC (control).
    default = int(config.round_trip_cost_bps)
    ranked = list(table["asset"]) if not table.empty else []
    detail_assets = ranked[:3] + (["BTC"] if "BTC" in ranked and "BTC" not in ranked[:3] else [])
    print("\nPer-asset detail (top 3 by transient opportunity + BTC control):")
    for asset in detail_assets:
        kdf, odf = basket_data[asset]
        a = analyze_dislocation(kdf, odf, config)
        print(format_asset_detail(asset, a, config))


if __name__ == "__main__":
    main()
