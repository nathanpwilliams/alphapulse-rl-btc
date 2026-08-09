"""
src/strategies/funding/analyzer.py - Funding-rate arbitrage analyzer (Phase A).

Diagnostic-only. This module answers ONE question from historical Kraken funding
data: does BTC perpetual funding-rate arbitrage clear realistic transaction costs?
It computes and prints statistics. There is deliberately NO execution, order, or
position-management logic here - interpretation and any trading decision happen
elsewhere.

The delta-neutral trade this analyzes is short-perp / long-spot: it RECEIVES
funding whenever funding is positive. The core accounting question is whether the
funding collected over a realistic holding period exceeds the all-in round-trip
transaction cost of opening and closing that pair.

Units are the whole game here. Kraken's `funding_rate` column is a PER-HOUR
proportional rate (e.g. 1e-5 == 0.001% per hour). Every conversion below is
annotated because per-hour-rate -> basis-points -> annualized is the easiest place
to introduce a silent factor-of-N bug.

    per-hour rate  -> bps:          rate * 10_000
    per-hour rate  -> annual %:     rate * 8760 * 100   (8760 = 24 * 365 hours/yr)

Public API:
    FundingAnalyzerConfig      dataclass of analysis parameters
    analyze_funding(df, cfg)   compute the full stats dict
    main()                     fetch live Kraken funding, analyze, print summary

Reuses `fetch_funding_rates` from src/data/load_kraken.py for data - funding
fetching is NOT reimplemented here.

CLI:
    python -m src.strategies.funding.analyzer
    python -m src.strategies.funding.analyzer --round-trip-bps 10
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.load_kraken import fetch_funding_rates

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

# Annualization factor. Funding is a per-hour rate; a year is 24 * 365 hours.
# (We use 365, not 365.25 - matches the spec and keeps the arithmetic auditable.)
HOURS_PER_YEAR = 24 * 365  # == 8760

# Basis-points-per-unit. A proportional rate of 1.0 == 10_000 bps.
BPS_PER_UNIT = 10_000

# Candidate holding periods for the net-after-cost table, in hours.
# 1d, 3d, 7d, 14d, 30d.
DEFAULT_HOLDING_PERIODS_HOURS = (24, 72, 168, 336, 720)


# ============================================================
# Config
# ============================================================

@dataclass
class FundingAnalyzerConfig:
    """Parameters controlling the funding-rate arbitrage analysis.

    Attributes:
        symbol: Kraken Futures perpetual symbol to analyze.
        round_trip_cost_bps: All-in round-trip transaction cost in basis points -
            both legs, entry + exit, including slippage. Parameterized so the cost
            assumption can be swept from the CLI.
        funding_period_hours: Settlement cadence of the funding rate, in hours.
            Kraken settles funding hourly, so the per-hour `funding_rate` column
            already IS the per-period rate (default 1.0).
        holding_periods_hours: Candidate holding periods for the
            net-after-cost-by-holding-period table.
    """

    symbol: str = "PF_XBTUSD"
    round_trip_cost_bps: float = 20.0
    funding_period_hours: float = 1.0
    holding_periods_hours: tuple[int, ...] = field(
        default_factory=lambda: DEFAULT_HOLDING_PERIODS_HOURS
    )


# ============================================================
# Core analysis
# ============================================================

def analyze_funding(
    funding_df: pd.DataFrame,
    config: FundingAnalyzerConfig,
) -> dict[str, Any]:
    """Compute funding-rate arbitrage diagnostic statistics.

    Args:
        funding_df: DataFrame with a DatetimeIndex and a `funding_rate` column of
            PER-HOUR proportional funding rates (the shape returned by
            `src.data.load_kraken.fetch_funding_rates`).
        config: analysis parameters (symbol, round-trip cost, holding periods).

    Returns:
        A dict of statistics grouped by concern. Top-level keys:
            coverage, sign_distribution, magnitude, gross_yield, cost_hurdle,
            sign_flips, net_by_holding_period, config.

    Raises:
        ValueError: if `funding_df` lacks a `funding_rate` column, has a
            non-DatetimeIndex, or is empty.
    """
    if "funding_rate" not in funding_df.columns:
        raise ValueError("funding_df must have a 'funding_rate' column")
    if not isinstance(funding_df.index, pd.DatetimeIndex):
        raise ValueError("funding_df must have a DatetimeIndex")
    if len(funding_df) == 0:
        raise ValueError("funding_df is empty; nothing to analyze")

    df = funding_df.sort_index()
    rate = df["funding_rate"].to_numpy(dtype=float)
    n = len(rate)

    # ---- 1. Coverage -------------------------------------------------------
    first_ts = df.index[0]
    last_ts = df.index[-1]
    # Wall-clock hours spanned by the observation window (endpoint - startpoint).
    total_hours_covered = (last_ts - first_ts).total_seconds() / 3600.0
    coverage = {
        "n_observations": int(n),
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "total_hours_covered": float(total_hours_covered),
    }

    # ---- 2. Sign distribution ---------------------------------------------
    n_pos = int(np.sum(rate > 0))
    n_neg = int(np.sum(rate < 0))
    n_zero = int(np.sum(rate == 0))
    sign_distribution = {
        "frac_positive": n_pos / n,
        "frac_negative": n_neg / n,
        "frac_zero": n_zero / n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_zero": n_zero,
    }

    # ---- 3. Magnitude ------------------------------------------------------
    mean_hourly = float(np.mean(rate))
    # mean_abs is the workhorse for cost/break-even math: a delta-neutral book can
    # position on EITHER side, so it earns the absolute funding regardless of sign.
    mean_abs_hourly = float(np.mean(np.abs(rate)))
    magnitude = {
        "mean_hourly": mean_hourly,
        "median_hourly": float(np.median(rate)),
        "std_hourly": float(np.std(rate)),
        "min_hourly": float(np.min(rate)),
        "max_hourly": float(np.max(rate)),
        "mean_abs_hourly": mean_abs_hourly,
        # per-hour rate -> annualized percent: rate * (24*365) * 100
        "mean_annualized_pct": mean_hourly * HOURS_PER_YEAR * 100,
    }

    # ---- 4. Directional-hold gross yield ----------------------------------
    # A delta-neutral short-perp/long-spot book receives +funding_rate per hour
    # when funding is positive. Held continuously on the receiving side, the naive
    # gross annualized yield is just the mean per-hour rate annualized.
    #   signed: uses mean(rate)     - the "always short perp" figure (can be < 0).
    #   abs:    uses mean(|rate|)   - the "always on the receiving side" figure.
    gross_yield = {
        "annualized_gross_pct": mean_hourly * HOURS_PER_YEAR * 100,
        "annualized_gross_abs_pct": mean_abs_hourly * HOURS_PER_YEAR * 100,
    }

    # ---- 5. Cost hurdle / break-even holding period -----------------------
    # How many hours must we hold before accumulated gross funding (at the mean
    # absolute per-hour rate) exceeds the round-trip cost?
    #   funding earned per hour, in bps = mean_abs_hourly * 10_000
    #   break_even_hours = round_trip_cost_bps / (funding per hour in bps)
    mean_abs_bps_per_hour = mean_abs_hourly * BPS_PER_UNIT  # rate -> bps
    if mean_abs_bps_per_hour > 0:
        break_even_hours: float = config.round_trip_cost_bps / mean_abs_bps_per_hour
    else:
        # Zero funding everywhere: the cost is never recovered.
        break_even_hours = float("inf")
    cost_hurdle = {
        "round_trip_cost_bps": config.round_trip_cost_bps,
        "mean_abs_bps_per_hour": mean_abs_bps_per_hour,
        "break_even_hours": break_even_hours,
        "break_even_days": break_even_hours / 24.0,
    }

    # ---- 6. Sign-flip / reversal frequency --------------------------------
    # A "flip" is the funding rate changing sign from one interval to the next.
    # High flip rate == positions get whipsawed before they mature. We compare the
    # sign of consecutive observations; zeros are treated as their own sign so a
    # +/0/+ sequence counts as two flips (a conservative view of instability).
    if n >= 2:
        signs = np.sign(rate)
        n_transitions = n - 1
        n_flips = int(np.sum(signs[1:] != signs[:-1]))
        flip_fraction = n_flips / n_transitions
        # Mean hours between flips: total hours / number of flips. If funding
        # settles hourly, one interval == funding_period_hours.
        span_hours = n_transitions * config.funding_period_hours
        mean_hours_between_flips = (
            span_hours / n_flips if n_flips > 0 else float("inf")
        )
    else:
        n_transitions = 0
        n_flips = 0
        flip_fraction = 0.0
        mean_hours_between_flips = float("inf")
    sign_flips = {
        "n_flips": n_flips,
        "n_transitions": n_transitions,
        "flip_fraction": flip_fraction,
        "mean_hours_between_flips": mean_hours_between_flips,
    }

    # ---- 7. Net-after-cost yield by holding period ------------------------
    # For each candidate holding period H: enter, hold H hours collecting the mean
    # absolute funding, then exit paying the full round-trip cost once.
    #   gross_bps = mean_abs_hourly * 10_000 * H        (rate -> bps, times H hours)
    #   net_bps   = gross_bps - round_trip_cost_bps     (one round trip per hold)
    # Annualized net yield if this enter/hold/exit cycle is repeated back-to-back:
    #   cycles_per_year   = HOURS_PER_YEAR / H
    #   annualized_net_pct = (net_bps / 10_000) * cycles_per_year * 100
    net_by_holding_period: dict[int, dict[str, float]] = {}
    for h in config.holding_periods_hours:
        gross_bps = mean_abs_hourly * BPS_PER_UNIT * h  # rate -> bps, over H hours
        net_bps = gross_bps - config.round_trip_cost_bps  # pay one round trip
        cycles_per_year = HOURS_PER_YEAR / h
        # net_bps / 10_000 converts bps back to a proportional per-cycle return.
        annualized_net_pct = (net_bps / BPS_PER_UNIT) * cycles_per_year * 100
        net_by_holding_period[int(h)] = {
            "gross_bps": float(gross_bps),
            "net_bps": float(net_bps),
            "annualized_net_pct": float(annualized_net_pct),
        }

    return {
        "coverage": coverage,
        "sign_distribution": sign_distribution,
        "magnitude": magnitude,
        "gross_yield": gross_yield,
        "cost_hurdle": cost_hurdle,
        "sign_flips": sign_flips,
        "net_by_holding_period": net_by_holding_period,
        "config": {
            "symbol": config.symbol,
            "round_trip_cost_bps": config.round_trip_cost_bps,
            "funding_period_hours": config.funding_period_hours,
        },
    }


# ============================================================
# Pretty printing
# ============================================================

def format_summary(stats: dict[str, Any]) -> str:
    """Render the analysis dict as a labeled, aligned, human-readable report."""
    cov = stats["coverage"]
    sign = stats["sign_distribution"]
    mag = stats["magnitude"]
    gross = stats["gross_yield"]
    hurdle = stats["cost_hurdle"]
    flips = stats["sign_flips"]
    net = stats["net_by_holding_period"]
    cfg = stats["config"]

    lines: list[str] = []
    bar = "=" * 64
    lines.append(bar)
    lines.append(f"  FUNDING-RATE ARBITRAGE ANALYZER (Phase A) - {cfg['symbol']}")
    lines.append(bar)

    lines.append("")
    lines.append("Coverage")
    lines.append("-" * 64)
    lines.append(f"  Observations         : {cov['n_observations']:,}")
    lines.append(f"  First timestamp      : {cov['first_timestamp']}")
    lines.append(f"  Last timestamp       : {cov['last_timestamp']}")
    lines.append(f"  Total hours covered  : {cov['total_hours_covered']:,.1f}"
                 f"  ({cov['total_hours_covered'] / 24:,.1f} days)")

    lines.append("")
    lines.append("Sign distribution")
    lines.append("-" * 64)
    lines.append(f"  Positive             : {sign['frac_positive'] * 100:6.2f}%"
                 f"  ({sign['n_positive']:,})")
    lines.append(f"  Negative             : {sign['frac_negative'] * 100:6.2f}%"
                 f"  ({sign['n_negative']:,})")
    lines.append(f"  Zero                 : {sign['frac_zero'] * 100:6.2f}%"
                 f"  ({sign['n_zero']:,})")

    lines.append("")
    lines.append("Magnitude (per-hour funding rate)")
    lines.append("-" * 64)
    lines.append(f"  Mean                 : {mag['mean_hourly']:+.8f}"
                 f"  ({mag['mean_hourly'] * BPS_PER_UNIT:+.4f} bps/hr)")
    lines.append(f"  Median               : {mag['median_hourly']:+.8f}")
    lines.append(f"  Std                  : {mag['std_hourly']:.8f}")
    lines.append(f"  Min                  : {mag['min_hourly']:+.8f}")
    lines.append(f"  Max                  : {mag['max_hourly']:+.8f}")
    lines.append(f"  Mean |rate|          : {mag['mean_abs_hourly']:.8f}"
                 f"  ({mag['mean_abs_hourly'] * BPS_PER_UNIT:.4f} bps/hr)")
    lines.append(f"  Mean annualized      : {mag['mean_annualized_pct']:+.2f}%")

    lines.append("")
    lines.append("Directional-hold gross yield (naive, always positioned)")
    lines.append("-" * 64)
    lines.append(f"  Signed (short perp)  : {gross['annualized_gross_pct']:+.2f}% / yr")
    lines.append(f"  Absolute (best side) : {gross['annualized_gross_abs_pct']:+.2f}% / yr")

    lines.append("")
    lines.append(f"Cost hurdle (round-trip cost = {hurdle['round_trip_cost_bps']:.1f} bps)")
    lines.append("-" * 64)
    lines.append(f"  Funding earned       : {hurdle['mean_abs_bps_per_hour']:.4f} bps/hr")
    lines.append(f"  Break-even hold      : {hurdle['break_even_hours']:,.1f} hours"
                 f"  ({hurdle['break_even_days']:,.2f} days)")

    lines.append("")
    lines.append("Sign-flip / reversal frequency")
    lines.append("-" * 64)
    lines.append(f"  Flip fraction        : {flips['flip_fraction'] * 100:6.2f}%"
                 f"  ({flips['n_flips']:,} of {flips['n_transitions']:,} intervals)")
    lines.append(f"  Mean hrs between flip: {flips['mean_hours_between_flips']:,.2f}")

    lines.append("")
    lines.append("Net-after-cost yield by holding period")
    lines.append("-" * 64)
    lines.append(f"  {'Hold':>6}  {'':>5}  {'gross':>10}  {'net':>10}  {'annualized':>12}")
    lines.append(f"  {'(hrs)':>6}  {'(d)':>5}  {'(bps)':>10}  {'(bps)':>10}  {'net (%/yr)':>12}")
    for h in sorted(net):
        row = net[h]
        lines.append(
            f"  {h:>6}  {h / 24:>5.0f}  {row['gross_bps']:>10.4f}"
            f"  {row['net_bps']:>+10.4f}  {row['annualized_net_pct']:>+12.2f}"
        )

    lines.append("")
    lines.append(bar)
    lines.append("  Diagnostic only. No viability conclusion is drawn here.")
    lines.append(bar)
    return "\n".join(lines)


# ============================================================
# CLI entrypoint
# ============================================================

def main(argv: list[str] | None = None) -> None:
    """Fetch live Kraken funding, run the analysis, and print the summary."""
    parser = argparse.ArgumentParser(
        description="Funding-rate arbitrage analyzer (Phase A, diagnostic only)."
    )
    parser.add_argument(
        "--symbol",
        default="PF_XBTUSD",
        help="Kraken Futures perpetual symbol (default: PF_XBTUSD).",
    )
    parser.add_argument(
        "--round-trip-bps",
        type=float,
        default=20.0,
        help="All-in round-trip transaction cost in basis points (default: 20.0).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = FundingAnalyzerConfig(
        symbol=args.symbol,
        round_trip_cost_bps=args.round_trip_bps,
    )

    logger.info("Fetching Kraken funding rates for %s ...", config.symbol)
    funding_df = fetch_funding_rates(config.symbol)

    stats = analyze_funding(funding_df, config)
    print(format_summary(stats))


if __name__ == "__main__":
    main()
