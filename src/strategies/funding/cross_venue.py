"""
src/strategies/funding/cross_venue.py - Cross-venue funding-spread analyzer.

Single-venue BTC funding arb on Kraken does not clear costs (funding sign flips
every ~10h; break-even ~70h even at 5bps). This module measures the alternative:
the funding SPREAD between two venues held delta-neutral. The question is whether
the Kraken-vs-OKX BTC funding spread is stable and wide enough to be worth
pursuing. Diagnostic only - it computes and prints statistics; it draws no
viability conclusion.

THE CADENCE NORMALIZATION (the #1 correctness risk):
    Kraken (PF_XBTUSD) settles funding HOURLY; its `funding_rate` is already a
    per-hour rate. OKX (BTC-USDT-SWAP) settles on a longer, DYNAMIC interval
    (currently 8h, historically 4h, escalating under stress); its published rate
    is a PERIOD rate. The OKX loader therefore exposes `funding_rate_hourly`
    (= period rate / inferred interval hours). Both venues must be on this common
    PER-HOUR basis before any spread is taken:

        spread_hourly = kraken_funding_hourly - okx_funding_hourly

    Subtracting an OKX period-rate from a Kraken hourly rate directly would be a
    correctness bug that invalidates every number below.

ALIGNMENT:
    The venues settle on different schedules and at different timestamps, so we
    reindex each onto a common HOURLY grid over their overlapping range, forward
    -filling each venue's most recent per-hour rate to each hour (a venue's
    funding applies until its next settlement). Forward-fill is backward-looking
    only - no look-ahead. A per-venue staleness tolerance caps how long a rate may
    be carried, so a data gap cannot silently propagate an ancient rate.

Reuses the venue-agnostic `_break_even` and `_net_by_holding_period` helpers from
the single-venue analyzer, feeding the spread's mean-absolute per-hour rate where
the single-venue path feeds one venue's. The cost math is thus identical by
construction.

CLI:
    python -m src.strategies.funding.cross_venue
    python -m src.strategies.funding.cross_venue --round-trip-bps 10
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.load_kraken import fetch_funding_rates
from src.data.load_okx import fetch_okx_funding_rates
from src.strategies.funding.analyzer import (
    BPS_PER_UNIT,
    DEFAULT_HOLDING_PERIODS_HOURS,
    HOURS_PER_YEAR,
    _break_even,
    _net_by_holding_period,
)

logger = logging.getLogger(__name__)


# ============================================================
# Alignment tolerances
# ============================================================
#
# How stale a venue's most-recent rate may be when carried forward to an hourly
# grid slot. Set just above each venue's settlement cadence so normal gaps fill
# but a genuine data outage leaves NaN (later dropped) instead of a stale rate.

# Kraken settles hourly; 2h absorbs minor jitter without bridging a real gap.
KRAKEN_STALENESS_TOL = pd.Timedelta(hours=2)
# OKX settles on a longer, dynamic interval (currently 8h). 12h covers the 8h
# cadence with slack while still catching a multi-settlement outage.
OKX_STALENESS_TOL = pd.Timedelta(hours=12)


# ============================================================
# Config
# ============================================================

@dataclass
class CrossVenueConfig:
    """Parameters for the cross-venue funding-spread analysis.

    Attributes:
        kraken_symbol: Kraken Futures perpetual symbol (per-hour funding).
        okx_inst_id: OKX instrument id (period funding, normalized to per-hour).
        round_trip_cost_bps: all-in round-trip transaction cost in basis points.
            For a cross-venue trade this should reflect BOTH venues' legs; it is
            left as one sweepable parameter, consistent with the single-venue
            analyzer.
        holding_periods_hours: holding-period grid for the net-after-cost table.
    """

    kraken_symbol: str = "PF_XBTUSD"
    okx_inst_id: str = "BTC-USDT-SWAP"
    round_trip_cost_bps: float = 20.0
    holding_periods_hours: tuple[int, ...] = field(
        default_factory=lambda: DEFAULT_HOLDING_PERIODS_HOURS
    )


# ============================================================
# Alignment
# ============================================================

def align_funding_series(
    kraken_df: pd.DataFrame,
    okx_df: pd.DataFrame,
    kraken_col: str = "funding_rate",
    okx_col: str = "funding_rate_hourly",
) -> pd.DataFrame:
    """Align two venues' PER-HOUR funding onto a common hourly grid.

    Both inputs must already be on a per-hour basis: Kraken's `funding_rate` is
    per-hour natively; OKX's `funding_rate_hourly` is its period rate divided by
    the inferred settlement interval. This function does NOT re-normalize - it
    assumes the caller passed per-hour columns.

    Each venue is reindexed onto an hourly grid spanning the two series' OVERLAP,
    forward-filling the most recent settlement to each hour (backward-looking, so
    no look-ahead) with a per-venue staleness tolerance. Hours where either venue
    has no rate within tolerance are dropped, so the result is fully populated.

    Args:
        kraken_df: DataFrame with DatetimeIndex and `kraken_col` (per-hour rate).
        okx_df: DataFrame with DatetimeIndex and `okx_col` (per-hour rate).
        kraken_col, okx_col: the per-hour columns to align.

    Returns:
        DataFrame indexed by the hourly grid with columns
        [kraken_hourly, okx_hourly, spread_hourly], covering only the overlap.

    Raises:
        ValueError: if inputs lack DatetimeIndexes/columns or do not overlap.
    """
    for name, df, col in (("kraken_df", kraken_df, kraken_col), ("okx_df", okx_df, okx_col)):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{name} must have a DatetimeIndex")
        if col not in df.columns:
            raise ValueError(f"{name} must have a {col!r} column")

    k = kraken_df[kraken_col].sort_index()
    o = okx_df[okx_col].sort_index()

    # Overlap only: start at the later of the two first timestamps, end at the
    # earlier of the two last. Outside this window one venue has no data.
    overlap_start = max(k.index[0], o.index[0])
    overlap_end = min(k.index[-1], o.index[-1])
    if overlap_start >= overlap_end:
        raise ValueError(
            f"Kraken and OKX funding do not overlap: "
            f"Kraken [{k.index[0]}..{k.index[-1]}], OKX [{o.index[0]}..{o.index[-1]}]."
        )

    # Common hourly grid. Anchor to whole hours so both venues ffill consistently.
    grid = pd.date_range(
        overlap_start.ceil("h"), overlap_end.floor("h"), freq="1h", tz="UTC"
    )

    # reindex(method="ffill", tolerance=...) takes, for each grid hour, the most
    # recent observation at or before it (no look-ahead), unless it is staler than
    # the tolerance (then NaN). This carries an 8h OKX rate across its 8 hours and
    # leaves Kraken's hourly rate essentially untouched.
    kraken_hourly = k.reindex(grid, method="ffill", tolerance=KRAKEN_STALENESS_TOL)
    okx_hourly = o.reindex(grid, method="ffill", tolerance=OKX_STALENESS_TOL)

    aligned = pd.DataFrame(
        {"kraken_hourly": kraken_hourly, "okx_hourly": okx_hourly}, index=grid
    )
    # Drop any hour missing a rate (grid edges / outages) so downstream math is
    # on a fully populated, apples-to-apples per-hour series.
    aligned = aligned.dropna()
    # Cross-venue per-hour spread. Both terms are per-hour, so this is meaningful.
    aligned["spread_hourly"] = aligned["kraken_hourly"] - aligned["okx_hourly"]
    return aligned


# ============================================================
# Core analysis
# ============================================================

def analyze_spread(
    kraken_df: pd.DataFrame,
    okx_df: pd.DataFrame,
    config: CrossVenueConfig,
) -> dict[str, Any]:
    """Compute cross-venue funding-spread diagnostic statistics.

    Args:
        kraken_df: Kraken funding (DatetimeIndex, `funding_rate` per-hour).
        okx_df: OKX funding (DatetimeIndex, `funding_rate_hourly` per-hour).
        config: analysis parameters (symbols, round-trip cost, holding grid).

    Returns:
        A dict grouped by concern. Top-level keys:
            coverage, spread, sign_stability, magnitude, cost_hurdle,
            net_by_holding_period, correlation, config.

    Raises:
        ValueError: if the two series cannot be aligned (no overlap, bad shape).
    """
    aligned = align_funding_series(kraken_df, okx_df)
    if len(aligned) < 2:
        raise ValueError("Aligned overlap has fewer than 2 hours; nothing to analyze")

    spread = aligned["spread_hourly"].to_numpy(dtype=float)
    kraken_h = aligned["kraken_hourly"].to_numpy(dtype=float)
    okx_h = aligned["okx_hourly"].to_numpy(dtype=float)
    n = len(spread)

    # ---- 1. Coverage -------------------------------------------------------
    coverage = {
        "overlap_start": aligned.index[0],
        "overlap_end": aligned.index[-1],
        "n_aligned_hours": int(n),
    }

    # ---- 2. Spread definition & magnitude ---------------------------------
    mean_spread = float(np.mean(spread))
    mean_abs_spread = float(np.mean(np.abs(spread)))  # drives all cost math below
    spread_stats = {
        "mean_hourly": mean_spread,
        "median_hourly": float(np.median(spread)),
        "std_hourly": float(np.std(spread)),
        "min_hourly": float(np.min(spread)),
        "max_hourly": float(np.max(spread)),
        # per-hour spread -> annualized percent: spread * (24*365) * 100
        "annualized_mean_pct": mean_spread * HOURS_PER_YEAR * 100,
    }

    # ---- 3. Spread sign stability (THE KEY METRIC) ------------------------
    # Cross-venue analogue of the single-venue sign-flip test. A spread that keeps
    # its sign for long stretches can actually be harvested; one that whipsaws
    # (like the single-venue level, ~10h between flips) cannot.
    n_pos = int(np.sum(spread > 0))
    n_neg = int(np.sum(spread < 0))
    n_zero = int(np.sum(spread == 0))
    signs = np.sign(spread)
    n_transitions = n - 1
    n_flips = int(np.sum(signs[1:] != signs[:-1]))
    flip_fraction = n_flips / n_transitions if n_transitions > 0 else 0.0
    # Grid is hourly, so one interval == one hour.
    mean_hours_between_flips = (
        n_transitions / n_flips if n_flips > 0 else float("inf")
    )
    sign_stability = {
        "frac_positive": n_pos / n,
        "frac_negative": n_neg / n,
        "frac_zero": n_zero / n,
        "n_flips": n_flips,
        "n_transitions": n_transitions,
        "flip_fraction": flip_fraction,
        "mean_hours_between_flips": mean_hours_between_flips,
    }

    # ---- 4. Harvestable magnitude -----------------------------------------
    magnitude = {
        "mean_abs_spread_hourly": mean_abs_spread,
        "mean_abs_spread_bps_per_hour": mean_abs_spread * BPS_PER_UNIT,  # rate->bps
    }

    # ---- 5 & 6. Break-even + net-after-cost (shared helpers) --------------
    # Feed the spread's mean-absolute per-hour rate into the exact helpers the
    # single-venue analyzer uses; the cost accounting is identical by construction.
    cost_hurdle = _break_even(mean_abs_spread, config.round_trip_cost_bps)
    net_by_holding_period = _net_by_holding_period(
        mean_abs_spread, config.round_trip_cost_bps, config.holding_periods_hours
    )

    # ---- 7. Correlation ---------------------------------------------------
    # Pearson correlation of the two venues' per-hour funding on the aligned grid.
    # High correlation + a small, stable spread => the venues move together and the
    # spread is the residual (structural), rather than two unrelated noisy levels.
    if np.std(kraken_h) > 0 and np.std(okx_h) > 0:
        correlation = float(np.corrcoef(kraken_h, okx_h)[0, 1])
    else:
        # A constant series has undefined correlation; report NaN rather than 0.
        correlation = float("nan")

    return {
        "coverage": coverage,
        "spread": spread_stats,
        "sign_stability": sign_stability,
        "magnitude": magnitude,
        "cost_hurdle": cost_hurdle,
        "net_by_holding_period": net_by_holding_period,
        "correlation": correlation,
        "config": {
            "kraken_symbol": config.kraken_symbol,
            "okx_inst_id": config.okx_inst_id,
            "round_trip_cost_bps": config.round_trip_cost_bps,
        },
    }


# ============================================================
# Pretty printing
# ============================================================

def format_summary(stats: dict[str, Any]) -> str:
    """Render the cross-venue analysis dict as a labeled, aligned report."""
    cov = stats["coverage"]
    sp = stats["spread"]
    sign = stats["sign_stability"]
    mag = stats["magnitude"]
    hurdle = stats["cost_hurdle"]
    net = stats["net_by_holding_period"]
    corr = stats["correlation"]
    cfg = stats["config"]

    lines: list[str] = []
    bar = "=" * 68
    lines.append(bar)
    lines.append("  CROSS-VENUE FUNDING SPREAD ANALYZER (Phase A)")
    lines.append(f"  Kraken {cfg['kraken_symbol']}  vs  OKX {cfg['okx_inst_id']}")
    lines.append(bar)

    lines.append("")
    lines.append("Coverage (overlapping hourly grid)")
    lines.append("-" * 68)
    lines.append(f"  Overlap start        : {cov['overlap_start']}")
    lines.append(f"  Overlap end          : {cov['overlap_end']}")
    lines.append(f"  Aligned hours        : {cov['n_aligned_hours']:,}"
                 f"  ({cov['n_aligned_hours'] / 24:,.1f} days)")

    lines.append("")
    lines.append("Spread = Kraken_hourly - OKX_hourly  (per-hour)")
    lines.append("-" * 68)
    lines.append(f"  Mean                 : {sp['mean_hourly']:+.8f}"
                 f"  ({sp['mean_hourly'] * BPS_PER_UNIT:+.4f} bps/hr)")
    lines.append(f"  Median               : {sp['median_hourly']:+.8f}")
    lines.append(f"  Std                  : {sp['std_hourly']:.8f}")
    lines.append(f"  Min                  : {sp['min_hourly']:+.8f}")
    lines.append(f"  Max                  : {sp['max_hourly']:+.8f}")
    lines.append(f"  Annualized mean      : {sp['annualized_mean_pct']:+.2f}%")

    lines.append("")
    lines.append("Spread sign stability  (KEY METRIC)")
    lines.append("-" * 68)
    lines.append(f"  Positive             : {sign['frac_positive'] * 100:6.2f}%")
    lines.append(f"  Negative             : {sign['frac_negative'] * 100:6.2f}%")
    lines.append(f"  Zero                 : {sign['frac_zero'] * 100:6.2f}%")
    lines.append(f"  Sign-flip fraction   : {sign['flip_fraction'] * 100:6.2f}%"
                 f"  ({sign['n_flips']:,} of {sign['n_transitions']:,} hrs)")
    lines.append(f"  Mean hrs between flip: {sign['mean_hours_between_flips']:,.2f}")

    lines.append("")
    lines.append("Harvestable spread magnitude")
    lines.append("-" * 68)
    lines.append(f"  Mean |spread|        : {mag['mean_abs_spread_hourly']:.8f}"
                 f"  ({mag['mean_abs_spread_bps_per_hour']:.4f} bps/hr)")

    lines.append("")
    lines.append(f"Cost hurdle (round-trip cost = {hurdle['round_trip_cost_bps']:.1f} bps)")
    lines.append("-" * 68)
    lines.append(f"  Break-even hold      : {hurdle['break_even_hours']:,.1f} hours"
                 f"  ({hurdle['break_even_days']:,.2f} days)")

    lines.append("")
    lines.append("Net-after-cost yield by holding period (spread)")
    lines.append("-" * 68)
    lines.append(f"  {'Hold':>6}  {'':>5}  {'gross':>10}  {'net':>10}  {'annualized':>12}")
    lines.append(f"  {'(hrs)':>6}  {'(d)':>5}  {'(bps)':>10}  {'(bps)':>10}  {'net (%/yr)':>12}")
    for h in sorted(net):
        row = net[h]
        lines.append(
            f"  {h:>6}  {h / 24:>5.0f}  {row['gross_bps']:>10.4f}"
            f"  {row['net_bps']:>+10.4f}  {row['annualized_net_pct']:>+12.2f}"
        )

    lines.append("")
    lines.append("Venue co-movement")
    lines.append("-" * 68)
    lines.append(f"  Pearson corr (K,OKX) : {corr:+.4f}")

    lines.append("")
    lines.append(bar)
    lines.append("  Diagnostic only. No viability conclusion is drawn here.")
    lines.append(bar)
    return "\n".join(lines)


# ============================================================
# CLI entrypoint
# ============================================================

def main(argv: list[str] | None = None) -> None:
    """Fetch both venues live, align, analyze the spread, and print the summary."""
    parser = argparse.ArgumentParser(
        description="Cross-venue funding-spread analyzer (Phase A, diagnostic only)."
    )
    parser.add_argument("--kraken-symbol", default="PF_XBTUSD",
                        help="Kraken Futures symbol (default: PF_XBTUSD).")
    parser.add_argument("--okx-inst-id", default="BTC-USDT-SWAP",
                        help="OKX instrument id (default: BTC-USDT-SWAP).")
    parser.add_argument("--round-trip-bps", type=float, default=20.0,
                        help="All-in round-trip transaction cost in bps (default: 20.0).")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = CrossVenueConfig(
        kraken_symbol=args.kraken_symbol,
        okx_inst_id=args.okx_inst_id,
        round_trip_cost_bps=args.round_trip_bps,
    )

    logger.info("Fetching Kraken funding for %s ...", config.kraken_symbol)
    kraken_df = fetch_funding_rates(config.kraken_symbol)
    logger.info("Fetching OKX funding for %s ...", config.okx_inst_id)
    okx_df = fetch_okx_funding_rates(config.okx_inst_id)

    stats = analyze_spread(kraken_df, okx_df, config)
    print(format_summary(stats))


if __name__ == "__main__":
    main()
