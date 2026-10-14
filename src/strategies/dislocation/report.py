"""
src/strategies/dislocation/report.py - Phase B executable-gap report.

Turns the chunked streaming driver's output (`streaming.analyze_streaming`) into
the Phase B result tables: coverage/clean hours, excursion counts and durations,
size-capped capacity, daily capturable dollars, hour-of-day and weekday/weekend
splits, direction split, and the Phase A candle-screen comparison.

This is the module that produced the committed Phase B numbers; it exists so the
headline figures are reproducible from the repo rather than from a scratch
script. Aggregation is pure (`list[record] -> dict`) and unit-tested; only
`collect()` touches disk.

Conventions that matter when reading the output
-----------------------------------------------
* **One record per excursion, read at its peak-net row.** `cap_usd`, `net_bps`
  and `direction` all come from the same row, so `capturable_usd` =
  cap_usd x net_bps / 10000 never multiplies a max capacity by an unrelated max
  edge. See `streaming` for the full convention.
* **Partial days are reported as rates.** The collection's first and last days
  are 22.6 h and 5.7 h, so a raw daily total understates them. Every per-day
  dollar/count figure is normalised: value / clean_hours x 24. Raw totals are
  printed alongside, never instead.
* **The ~1 s Kraken resolution floor is reported separately.** Kraken's `ticker`
  feed delivers ~1 update/s/asset against OKX's tens, so an "excursion" shorter
  than ~1 s may be a stale-quote artifact rather than a dislocation. Durations
  are reported with the <=1.0 s fraction broken out, and the Phase A comparison
  is repeated on the >1 s subset.
* **Outage windows are masked, not interpolated.** `mask_windows` excludes
  collector/feed outages from every metric and from the active-time denominator.

Diagnostic only. No orders.

CLI:
    python -m src.strategies.dislocation.report --dir data/tob
    python -m src.strategies.dislocation.report --dir data/tob --json-out out.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.strategies.dislocation.executable import ExecutableConfig
from src.strategies.dislocation.streaming import StreamConfig, analyze_streaming

logger = logging.getLogger(__name__)


# ============================================================
# Config
# ============================================================

# The 396.6 s Kraken HTTP-503 outage found in the day-11 health check of the
# 2026-10-02 -> 2026-10-13 collection. Masked from every metric.
PHASE_B_MASK: list[tuple[str, str]] = [
    ("2026-10-06 07:00:42", "2026-10-06 07:07:19"),
]

# Phase A screen, 30d of 5m candles at a 20 bps hurdle on mean-removed gaps
# (transient excursions/day). Not directly comparable to Phase B's net>0 after
# 10 bps on raw gaps -- the ratio reflects threshold and mean-removal
# differences as well as the resolution change. See docs/journal.
PHASE_A_EXCURSIONS_PER_DAY: dict[str, float] = {
    "BTC": 0.1, "UNI": 7.4, "NEAR": 17.9, "ARB": 41.9,
}

# UTC days in the Phase B collection that are not a full 24 h.
PHASE_B_PARTIAL_DAYS = frozenset({"20261002", "20261013"})

# The ~1 s Kraken `ticker` cadence. Excursions at or below this are reported
# separately; they cannot be resolved by this dataset.
RESOLUTION_FLOOR_SEC = 1.0


@dataclass
class ReportConfig:
    """Parameters for the Phase B report.

    Attributes:
        assets: assets to report, control first by convention (BTC).
        mask_windows: (start, end) UTC strings excluded from all metrics.
        fee_bps_per_side: taker fee per leg; both legs are taker.
        chunksize: rows per read_csv chunk in the streaming driver.
        phase_a: Phase A candle-screen excursions/day, per asset.
        partial_days: YYYYMMDD days shorter than 24 h (marked in the tables).
        resolution_floor_sec: duration at/below which an excursion is treated as
            unresolvable by this dataset.
    """

    assets: Sequence[str] = ("BTC", "ARB", "NEAR", "UNI")
    mask_windows: Sequence[tuple[str, str]] = tuple(PHASE_B_MASK)
    fee_bps_per_side: float = 5.0
    chunksize: int = 1_000_000
    phase_a: dict[str, float] = field(default_factory=lambda: dict(PHASE_A_EXCURSIONS_PER_DAY))
    partial_days: frozenset[str] = PHASE_B_PARTIAL_DAYS
    resolution_floor_sec: float = RESOLUTION_FLOOR_SEC

    def stream_config(self) -> StreamConfig:
        """The streaming-driver config implied by this report config."""
        return StreamConfig(
            chunksize=self.chunksize,
            mask_windows=[(pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"))
                          for a, b in self.mask_windows],
            executable=ExecutableConfig(fee_bps_per_side=self.fee_bps_per_side,
                                        kraken_fee_bps=self.fee_bps_per_side,
                                        okx_fee_bps=self.fee_bps_per_side),
        )


# ============================================================
# Aggregation helpers (pure)
# ============================================================

def _p(a: np.ndarray, q: float) -> float:
    return float(np.nanpercentile(a, q)) if len(a) else float("nan")


def is_weekend(day: str) -> bool:
    """True if 'YYYYMMDD' is a Saturday or Sunday (UTC)."""
    return pd.Timestamp(day).weekday() >= 5


def clean_hours(per_day: dict[str, dict[str, Any]]) -> float:
    """Total outage- and mask-excluded hours across all days."""
    return sum(d["active_seconds"] for d in per_day.values()) / 3600.0


def rate_per_day(value: float, active_seconds: float) -> float:
    """Normalise a per-day total to a 24 h rate so partial days compare."""
    if active_seconds <= 0:
        return float("nan")
    return value / (active_seconds / 3600.0) * 24.0


def coverage(per_day: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Aligned-update count and covered/outage/masked/clean hours."""
    return {
        "n_aligned": sum(d["n_aligned"] for d in per_day.values()),
        "covered_hours": sum(d["covered_seconds"] for d in per_day.values()) / 3600.0,
        "outage_hours": sum(d["outage_seconds"] for d in per_day.values()) / 3600.0,
        "masked_seconds": sum(d.get("masked_seconds", 0.0) for d in per_day.values()),
        "clean_hours": clean_hours(per_day),
        "n_outages": sum(d["n_outages"] for d in per_day.values()),
    }


def duration_stats(records: Iterable[dict[str, Any]], floor_sec: float = RESOLUTION_FLOOR_SEC
                   ) -> dict[str, float]:
    """Duration percentiles plus the fraction at/below the resolution floor."""
    d = np.array([r["duration_sec"] for r in records], dtype=float)
    if not len(d):
        return {"n": 0}
    return {
        "n": int(len(d)),
        "p10": _p(d, 10), "median": float(np.median(d)), "p75": _p(d, 75),
        "p90": _p(d, 90), "p99": _p(d, 99), "max": float(d.max()),
        "n_le_floor": int((d <= floor_sec).sum()),
        "n_gt_floor": int((d > floor_sec).sum()),
        "n_gt_5s": int((d > 5).sum()),
        "n_gt_60s": int((d > 60).sum()),
        "frac_le_floor": float((d <= floor_sec).mean()),
    }


def capacity_stats(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Size-capped USD capacity per excursion: median/p90/mean/max.

    Sizes are converted to base units per venue before the min() that produces
    `cap_usd` (OKX quotes contracts); `frac_okx_binds` reports how often OKX's
    depth was the binding side, which is the diagnostic for that conversion.
    """
    recs = list(records)
    c = np.array([r["cap_usd"] for r in recs], dtype=float)
    if not len(c):
        return {"n": 0}
    binds = [r["okx_leg_binds"] for r in recs if "okx_leg_binds" in r]
    out = {"n": int(len(c)), "median": float(np.nanmedian(c)), "p90": _p(c, 90),
           "mean": float(np.nanmean(c)), "max": float(np.nanmax(c))}
    if binds:
        out["frac_okx_binds"] = float(np.mean(binds))
    return out


def minority_direction(records: Iterable[dict[str, Any]]) -> str | None:
    """The less common crossing direction - the only mean-reverting (tradeable) part.

    A persistent one-sided split is a structural premium with no closing leg; the
    minority side is what an arbitrageur could actually capture.
    """
    recs = list(records)
    if not recs:
        return None
    d = direction_split(recs)
    a, b = d["sell_k_buy_o"], d["sell_o_buy_k"]
    return "sell_o_buy_k" if b["n"] <= a["n"] else "sell_k_buy_o"


def filtered_views(
    records: Iterable[dict[str, Any]], clean_h: float,
    floor_sec: float = RESOLUTION_FLOOR_SEC,
) -> dict[str, dict[str, float]]:
    """The three committed views: all, >floor, and minority-direction & >floor.

    Each reports excursion count, per-day rate, capacity percentiles and
    rate-normalised capturable dollars, so the headline can be read at the
    successively stricter standards without re-deriving them by hand.
    """
    recs = list(records)
    minority = minority_direction(recs)
    views = {
        "all": recs,
        "above_floor": [r for r in recs if r["duration_sec"] > floor_sec],
        "transient_above_floor": [r for r in recs
                                  if r["duration_sec"] > floor_sec
                                  and r["direction"] == minority],
    }
    out: dict[str, dict[str, float]] = {}
    for name, sl in views.items():
        usd = float(np.nansum([r["capturable_usd"] for r in sl])) if sl else 0.0
        cap = capacity_stats(sl)
        out[name] = {
            "n": len(sl),
            "per_day": rate_per_day(len(sl), clean_h * 3600.0),
            "total_usd": usd,
            "usd_per_day": rate_per_day(usd, clean_h * 3600.0),
            "median_cap_usd": cap.get("median", float("nan")),
            "p90_cap_usd": cap.get("p90", float("nan")),
            "max_cap_usd": cap.get("max", float("nan")),
        }
    out["minority_direction"] = minority
    return out


def net_bps_stats(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Net bps per excursion (at its peak-net row): median/p90/max."""
    b = np.array([r["net_bps"] for r in records], dtype=float)
    if not len(b):
        return {"n": 0}
    return {"n": int(len(b)), "median": float(np.median(b)), "p90": _p(b, 90),
            "max": float(b.max())}


def daily_table(per_day: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-day raw and rate-normalised excursion counts and capturable dollars."""
    out = {}
    for day, s in sorted(per_day.items()):
        out[day] = {
            "n_excursions": s["n_excursions"],
            "capturable_usd": s["capturable_usd"],
            "clean_hours": s["active_seconds"] / 3600.0,
            "usd_per_day": rate_per_day(s["capturable_usd"], s["active_seconds"]),
            "exc_per_day": rate_per_day(s["n_excursions"], s["active_seconds"]),
        }
    return out


def daily_summary(daily: dict[str, dict[str, float]], key: str) -> dict[str, float]:
    """Mean/min/max/median of a rate-normalised per-day series."""
    v = np.array([d[key] for d in daily.values()], dtype=float)
    if not len(v):
        return {}
    return {"mean": float(np.nanmean(v)), "min": float(np.nanmin(v)),
            "max": float(np.nanmax(v)), "median": float(np.nanmedian(v))}


def hour_histogram(records: Iterable[dict[str, Any]]) -> list[dict[str, float]]:
    """Excursion counts and capturable dollars by UTC hour-of-day (24 rows)."""
    recs = list(records)
    rows = []
    for h in range(24):
        sl = [r for r in recs if r["hour"] == h]
        rows.append({
            "hour": h,
            "n": len(sl),
            "usd": float(np.nansum([r["capturable_usd"] for r in sl])) if sl else 0.0,
        })
    return rows


def weekday_weekend(per_day: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Excursion counts and rate-normalised dollars, split weekday vs weekend."""
    out = {}
    for label, want in (("weekday", False), ("weekend", True)):
        days = [d for d in per_day if is_weekend(d) == want]
        secs = sum(per_day[d]["active_seconds"] for d in days)
        usd = sum(per_day[d]["capturable_usd"] for d in days)
        out[label] = {
            "n_excursions": sum(per_day[d]["n_excursions"] for d in days),
            "clean_hours": secs / 3600.0,
            "usd_per_day": rate_per_day(usd, secs),
        }
    wd, we = out["weekday"]["usd_per_day"], out["weekend"]["usd_per_day"]
    out["ratio_we_wd"] = we / wd if wd else float("nan")
    return out


def direction_split(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Counts, dollars and median net bps per crossing direction.

    A persistent one-sided split is the signature of a structural (quote-currency)
    premium rather than a transient dislocation: a premium has no closing leg.
    """
    recs = list(records)
    if not recs:
        return {"n": 0}
    out: dict[str, Any] = {"n": len(recs)}
    for d in ("sell_k_buy_o", "sell_o_buy_k"):
        sl = [r for r in recs if r["direction"] == d]
        out[d] = {
            "n": len(sl),
            "frac": len(sl) / len(recs),
            "usd": float(np.nansum([r["capturable_usd"] for r in sl])) if sl else 0.0,
            "median_net_bps": float(np.median([r["net_bps"] for r in sl])) if sl else float("nan"),
        }
    return out


def phase_a_comparison(
    records: Iterable[dict[str, Any]], clean_h: float, candle_per_day: float,
    floor_sec: float = RESOLUTION_FLOOR_SEC,
) -> dict[str, float]:
    """Phase A candle excursions/day vs Phase B executable, all and above the floor."""
    recs = list(records)
    above = [r for r in recs if r["duration_sec"] > floor_sec]
    exc = rate_per_day(len(recs), clean_h * 3600.0)
    exc_above = rate_per_day(len(above), clean_h * 3600.0)
    usd_above = float(np.nansum([r["capturable_usd"] for r in above])) if above else 0.0
    return {
        "candle_per_day": candle_per_day,
        "executable_per_day": exc,
        "ratio": exc / candle_per_day if candle_per_day > 0 else float("nan"),
        "n_above_floor": len(above),
        "executable_per_day_above_floor": exc_above,
        "usd_per_day_above_floor": rate_per_day(usd_above, clean_h * 3600.0),
        "ratio_above_floor": exc_above / candle_per_day if candle_per_day > 0 else float("nan"),
    }


def summarize(results: dict[str, dict[str, Any]], config: ReportConfig) -> dict[str, Any]:
    """Full aggregation for every asset. `results[asset]` is `analyze_streaming` output."""
    summary: dict[str, Any] = {}
    for asset, res in results.items():
        per_day, recs = res["per_day"], res["records"]
        ch = clean_hours(per_day)
        daily = daily_table(per_day)
        summary[asset] = {
            "coverage": coverage(per_day),
            "n_excursions": len(recs),
            "exc_per_day": rate_per_day(len(recs), ch * 3600.0),
            "best_net_bps": max((d["best_net_bps"] for d in per_day.values()),
                                default=float("nan")),
            "total_capturable_usd": float(np.nansum([r["capturable_usd"] for r in recs]))
                                    if recs else 0.0,
            "durations": duration_stats(recs, config.resolution_floor_sec),
            "capacity": capacity_stats(recs),
            "views": filtered_views(recs, ch, config.resolution_floor_sec),
            "net_bps": net_bps_stats(recs),
            "daily": daily,
            "usd_per_day": daily_summary(daily, "usd_per_day"),
            "exc_per_day_daily": daily_summary(daily, "exc_per_day"),
            "hours": hour_histogram(recs),
            "week": weekday_weekend(per_day),
            "direction": direction_split(recs),
            "phase_a": phase_a_comparison(recs, ch, config.phase_a.get(asset, float("nan")),
                                          config.resolution_floor_sec),
        }
    return summary


# ============================================================
# Collection
# ============================================================

def collect(out_dir: Path | str, config: ReportConfig) -> dict[str, dict[str, Any]]:
    """Run the streaming driver for every configured asset (the only disk-touching step)."""
    sc = config.stream_config()
    results = {}
    for asset in config.assets:
        results[asset] = analyze_streaming(out_dir, asset, sc)
        logger.info("%s: %d excursions", asset, len(results[asset]["records"]))
    return results


# ============================================================
# Formatting
# ============================================================

def _fmt_assets(assets: Sequence[str]) -> str:
    return "".join(f"{a:>12}" for a in assets)


def format_report(summary: dict[str, Any], config: ReportConfig) -> str:
    """The Phase B tables as text. Numbers only; no verdict."""
    assets = [a for a in config.assets if a in summary]
    L: list[str] = []
    w = 100

    L.append("=" * w)
    L.append("  PHASE B - EXECUTABLE CROSS-VENUE GAP, FULL COLLECTION")
    masks = ", ".join(f"{a} -> {b} UTC" for a, b in config.mask_windows)
    L.append(f"  masked: {masks or 'none'}")
    L.append(f"  fees: {config.fee_bps_per_side} + {config.fee_bps_per_side} "
             f"= {2 * config.fee_bps_per_side:.0f} bps round trip (taker both legs)")
    L.append(f"  partial days ({', '.join(sorted(config.partial_days))}) marked '*'; "
             f"per-day figures are rates: value / clean_h x 24")
    L.append("=" * w)

    # -------- coverage
    L.append("\n### COVERAGE / CLEAN HOURS\n")
    L.append(f"{'asset':<7}{'aligned updates':>18}{'covered h':>11}{'outage h':>10}"
             f"{'masked s':>10}{'CLEAN h':>10}{'outages':>9}")
    for a in assets:
        c = summary[a]["coverage"]
        L.append(f"{a:<7}{c['n_aligned']:>18,}{c['covered_hours']:>11.2f}"
                 f"{c['outage_hours']:>10.2f}{c['masked_seconds']:>10.1f}"
                 f"{c['clean_hours']:>10.2f}{c['n_outages']:>9,}")

    # -------- headline
    L.append("\n### HEADLINE\n")
    L.append(f"{'asset':<7}{'excursions':>12}{'exc/day':>10}{'best net bps':>14}"
             f"{'med dur s':>11}{'p90 dur s':>11}{'max dur s':>11}{'frac<=1s':>10}")
    for a in assets:
        s, d = summary[a], summary[a]["durations"]
        if not d.get("n"):
            L.append(f"{a:<7}{0:>12}{0.0:>10.2f}{s['best_net_bps']:>14.2f}")
            continue
        L.append(f"{a:<7}{s['n_excursions']:>12,}{s['exc_per_day']:>10.2f}"
                 f"{s['best_net_bps']:>14.2f}{d['median']:>11.3f}{d['p90']:>11.3f}"
                 f"{d['max']:>11.2f}{d['frac_le_floor'] * 100:>9.1f}%")

    # -------- durations
    L.append(f"\n### DURATION DETAIL (s) - <={config.resolution_floor_sec:g}s is at/below "
             f"the Kraken ticker floor\n")
    L.append(f"{'asset':<7}{'n':>9}{'p10':>9}{'median':>9}{'p75':>9}{'p90':>9}{'p99':>9}"
             f"{'max':>9}{'<=1.0s':>10}{'>1.0s':>9}{'>5s':>8}{'>60s':>7}")
    for a in assets:
        d = summary[a]["durations"]
        if not d.get("n"):
            L.append(f"{a:<7}{0:>9}")
            continue
        L.append(f"{a:<7}{d['n']:>9,}{d['p10']:>9.3f}{d['median']:>9.3f}{d['p75']:>9.3f}"
                 f"{d['p90']:>9.3f}{d['p99']:>9.2f}{d['max']:>9.2f}"
                 f"{d['n_le_floor']:>10,}{d['n_gt_floor']:>9,}{d['n_gt_5s']:>8,}"
                 f"{d['n_gt_60s']:>7,}")

    # -------- capacity
    L.append("\n### SIZE-CAPPED CAPACITY PER EXCURSION (USD, min(bid_size,ask_size) x mid "
             "at the peak-net row)\n")
    L.append(f"{'asset':<7}{'median $':>12}{'p90 $':>12}{'mean $':>12}{'max $':>14}"
             f"{'OKX leg binds':>15}")
    for a in assets:
        c = summary[a]["capacity"]
        if not c.get("n"):
            L.append(f"{a:<7}{'-':>12}")
            continue
        L.append(f"{a:<7}{c['median']:>12,.0f}{c['p90']:>12,.0f}{c['mean']:>12,.0f}"
                 f"{c['max']:>14,.0f}{c.get('frac_okx_binds', float('nan')) * 100:>14.1f}%")

    # -------- the three committed views
    L.append("\n### VIEWS: all / above the resolution floor / minority direction & above floor\n")
    L.append(f"{'asset':<7}{'view':<24}{'n':>9}{'per day':>10}{'med cap $':>11}"
             f"{'p90 cap $':>11}{'total $':>11}{'$/day':>9}")
    for a in assets:
        v = summary[a].get("views", {})
        for name in ("all", "above_floor", "transient_above_floor"):
            e = v.get(name)
            if e is None:
                continue
            label = name if name != "transient_above_floor" else \
                f"transient ({v.get('minority_direction') or '-'})"
            L.append(f"{a:<7}{label:<24}{e['n']:>9,}{e['per_day']:>10.2f}"
                     f"{e['median_cap_usd']:>11,.0f}{e['p90_cap_usd']:>11,.0f}"
                     f"{e['total_usd']:>11,.0f}{e['usd_per_day']:>9,.0f}")

    # -------- dollars
    L.append("\n### DAILY CAPTURABLE DOLLARS  "
             "sum( min(bid_size,ask_size) x mid x net_bps / 10000 )   [HEADLINE]\n")
    days = sorted({d for a in assets for d in summary[a]["daily"]})
    L.append(f"{'day':<11}{'dow':<5}" + "".join(f"{a + ' $/d':>13}" for a in assets)
             + f"{'clean h':>9}")
    for day in days:
        mark = "*" if day in config.partial_days else " "
        dow = pd.Timestamp(day).strftime("%a")
        line = f"{day[4:6]}-{day[6:8]}{mark:<6}{dow:<5}"
        ch = float("nan")
        for a in assets:
            e = summary[a]["daily"].get(day)
            line += f"{e['usd_per_day'] if e else float('nan'):>13,.0f}"
            if e:
                ch = e["clean_hours"]
        L.append(line + f"{ch:>9.2f}")

    L.append(f"\n{'asset':<7}{'total raw $':>14}{'mean $/day':>13}{'min $/day':>12}"
             f"{'max $/day':>12}{'median $/day':>14}   (rate-normalised)")
    for a in assets:
        s = summary[a]
        u = s["usd_per_day"]
        L.append(f"{a:<7}{s['total_capturable_usd']:>14,.0f}{u['mean']:>13,.0f}"
                 f"{u['min']:>12,.0f}{u['max']:>12,.0f}{u['median']:>14,.0f}")

    L.append(f"\n{'asset':<7}{'exc/day mean':>14}{'min':>9}{'max':>9}   "
             f"(rate-normalised per-day counts)")
    for a in assets:
        e = summary[a]["exc_per_day_daily"]
        L.append(f"{a:<7}{e['mean']:>14.2f}{e['min']:>9.2f}{e['max']:>9.2f}")

    L.append(f"\n{'asset':<7}" + "".join(f"{d[4:6] + '-' + d[6:8]:>8}" for d in days)
             + "    raw excursion count/day")
    for a in assets:
        L.append(f"{a:<7}" + "".join(
            f"{summary[a]['daily'].get(d, {}).get('n_excursions', 0):>8,}" for d in days))

    # -------- hour of day
    L.append("\n### HOUR-OF-DAY (UTC): excursion counts and capturable dollars\n")
    L.append(f"{'hr':<4}" + "".join(f"{a + ' n':>9}{a + ' $':>11}" for a in assets))
    for h in range(24):
        line = f"{h:02d}  "
        for a in assets:
            row = summary[a]["hours"][h]
            line += f"{row['n']:>9,}{row['usd']:>11,.0f}"
        L.append(line)

    # -------- weekday/weekend
    L.append("\n### WEEKDAY vs WEEKEND\n")
    L.append(f"{'asset':<7}{'wd exc':>9}{'wd $/day':>12}{'we exc':>9}{'we $/day':>12}"
             f"{'we/wd $':>10}")
    for a in assets:
        wk = summary[a]["week"]
        L.append(f"{a:<7}{wk['weekday']['n_excursions']:>9,}"
                 f"{wk['weekday']['usd_per_day']:>12,.0f}"
                 f"{wk['weekend']['n_excursions']:>9,}"
                 f"{wk['weekend']['usd_per_day']:>12,.0f}{wk['ratio_we_wd']:>10.2f}")

    # -------- direction
    L.append("\n### DIRECTION SPLIT  (a persistent one-sided split = structural premium, "
             "not a dislocation)\n")
    L.append(f"{'asset':<7}{'sellK/buyO n':>14}{'%':>7}{'$':>13}{'sellO/buyK n':>14}"
             f"{'%':>7}{'$':>13}{'med bps K>O':>13}{'med bps O>K':>13}")
    for a in assets:
        d = summary[a]["direction"]
        if not d.get("n"):
            L.append(f"{a:<7}{'-':>14}")
            continue
        A, B = d["sell_k_buy_o"], d["sell_o_buy_k"]
        L.append(f"{a:<7}{A['n']:>14,}{A['frac'] * 100:>6.1f}%{A['usd']:>13,.0f}"
                 f"{B['n']:>14,}{B['frac'] * 100:>6.1f}%{B['usd']:>13,.0f}"
                 f"{A['median_net_bps']:>13.2f}{B['median_net_bps']:>13.2f}")

    # -------- net bps
    L.append("\n### NET BPS PER EXCURSION (at the peak-net row)\n")
    L.append(f"{'asset':<7}{'median':>9}{'p90':>9}{'max':>9}")
    for a in assets:
        b = summary[a]["net_bps"]
        if not b.get("n"):
            L.append(f"{a:<7}{'-':>9}")
            continue
        L.append(f"{a:<7}{b['median']:>9.2f}{b['p90']:>9.2f}{b['max']:>9.2f}")

    # -------- phase A
    L.append("\n### PHASE A (5m candles, 20 bps hurdle) vs PHASE B EXECUTABLE (top-of-book, "
             "net of fees)\n")
    L.append(f"{'asset':<7}{'candle exc/day':>16}{'executable exc/day':>20}{'ratio B/A':>12}")
    for a in assets:
        p = summary[a]["phase_a"]
        L.append(f"{a:<7}{p['candle_per_day']:>16.1f}{p['executable_per_day']:>20.2f}"
                 f"{p['ratio']:>12.2f}")

    L.append(f"\n### SAME, ONLY EXCURSIONS LONGER THAN THE "
             f"{config.resolution_floor_sec:g} s KRAKEN FLOOR\n")
    L.append(f"{'asset':<7}{'exc >1s':>10}{'exc/day >1s':>14}{'$/day >1s':>12}{'ratio B/A':>12}")
    for a in assets:
        p = summary[a]["phase_a"]
        L.append(f"{a:<7}{p['n_above_floor']:>10,}{p['executable_per_day_above_floor']:>14.2f}"
                 f"{p['usd_per_day_above_floor']:>12,.0f}{p['ratio_above_floor']:>12.2f}")

    L.append("\n" + "=" * w)
    L.append("  Diagnostic only. No orders. Numbers only - interpretation is the lead's.")
    L.append("=" * w)
    return "\n".join(L)


# ============================================================
# CLI
# ============================================================

def _json_default(o: Any) -> Any:
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, frozenset):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase B executable-gap report.")
    parser.add_argument("--dir", default="data/tob", help="Directory of collected tob_*.csv.")
    parser.add_argument("--assets", default="BTC,ARB,NEAR,UNI",
                        help="Comma-separated; control (BTC) first by convention.")
    parser.add_argument("--fee-bps-per-side", type=float, default=5.0)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--mask", action="append", default=None,
                        help="UTC window to exclude, 'START/END'. Repeatable. "
                             "Omit to use the Phase B default outage mask.")
    parser.add_argument("--no-mask", action="store_true", help="Disable all masking.")
    parser.add_argument("--json-out", default=None,
                        help="Also write the full summary (incl. per-excursion records) here.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.no_mask:
        windows: list[tuple[str, str]] = []
    elif args.mask:
        windows = [tuple(w.split("/", 1)) for w in args.mask]  # type: ignore[misc]
    else:
        windows = list(PHASE_B_MASK)

    config = ReportConfig(
        assets=[a.strip().upper() for a in args.assets.split(",")],
        mask_windows=windows,
        fee_bps_per_side=args.fee_bps_per_side,
        chunksize=args.chunksize,
    )

    results = collect(args.dir, config)
    summary = summarize(results, config)
    print(format_report(summary, config))

    if args.json_out:
        payload = {"summary": summary,
                   "records": {a: results[a]["records"] for a in results}}
        with open(args.json_out, "w") as f:
            json.dump(payload, f, default=_json_default)
        logger.info("Wrote %s", args.json_out)


if __name__ == "__main__":
    main()
