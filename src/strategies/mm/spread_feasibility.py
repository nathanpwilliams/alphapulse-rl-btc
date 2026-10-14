"""
src/strategies/mm/spread_feasibility.py - Passive spread-capture feasibility (Path B).

Phase B rejected cross-venue arbitrage: the thin-alt bid-ask spread that made
cross-venue gaps uncapturable is the one income source on the original strategy
list never measured. This module estimates PASSIVE market-making economics for
NEAR and UNI (ARB as a low-liquidity extra, BTC as control) from the 11-day
top-of-book set already in `data/tob/`. No new collection, no execution logic,
no verdict - it computes and reports.

WHAT THIS CANNOT DO (the binding limitation)
--------------------------------------------
The dataset is QUOTES ONLY - there is no trade tape. Quote data gives spread
income per fill and adverse-selection cost per fill; it CANNOT give fill rates.
So everything here is **per-fill economics and ratios, never daily dollars**.
A number like "$X/day from market making" is not derivable from this dataset and
is deliberately absent. The headline is the dimensionless `edge_ratio`.

A second, related limit: a quote leaving the book is NOT evidence of a fill - it
could equally be a cancel, and on a 1 s snapshot feed it could be neither
(just an unobserved intermediate state). Fills are therefore only inferred from
**trade-throughs**: the one case where a resting order must have been hit is when
the opposite side of the book moves through its price. Nothing else is treated as
a fill.

Method
------
Per asset x venue, quotes are resampled onto a regular 1 s grid (backward as-of
with a staleness tolerance; never forward-filled across an outage) and the whole
collection's grid is concatenated so horizon windows span midnight correctly.
On that grid:

* **Spread** (ask - bid)/mid in bps: median, p10/p90, by hour-of-day and
  weekday/weekend, plus the fraction of TIME the spread is too tight to make.
* **Net half-spread** = median spread / 2 - maker fee per side. Gross income per
  one-sided fill.
* **Realized mid volatility** sigma(tau) = std of log(mid_{t+tau}/mid_t) in bps.
* **Adverse markout** from a naive touch quoter: at each grid point assume a
  resting bid at the current best bid and ask at the current best ask. An adverse
  event on the bid is `min(best_ask over (t, t+tau]) <= resting_bid` - the market
  traded down through the level, so a fill is certain and adverse. Symmetric for
  the ask. Reported as the event rate, and the mean markout over those events.
* **Edge ratio** = net half-spread / mean adverse cost, per tau. THE HEADLINE.
* **Queue context**: USD size resting at the touch (what a new quote joins
  behind).
* **Inventory drift proxy**: 5-minute mid-return distribution - what one-sided
  exposure costs over a repricing cycle.

Resolution caveat: Kraken's `ticker` feed is a ~1 update/s/asset SNAPSHOT, so any
Kraken figure at tau = 1 s sits on the resolution floor and understates true
short-horizon volatility and adverse selection (the intermediate ticks simply
were not delivered). OKX's `bbo-tbt` is genuinely tick-by-tick and is the cleaner
view of short-horizon dynamics. Both are reported; Kraken tau=1 s is flagged.

Diagnostic only. No orders.

CLI:
    python -m src.strategies.mm.spread_feasibility --dir data/tob
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.strategies.dislocation.report import PHASE_B_MASK
from src.strategies.dislocation.streaming import (
    QUOTE_COLS,
    StreamConfig,
    daily_files,
    load_asset_day,
)

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

# Published MAKER fees, perpetuals, base/lowest tier, checked 2026-10-14:
#   Kraken Futures  Tier 1 (<$5M 30d futures volume): maker 0.020% -> 2.0 bps
#                   (taker 0.050%, matching collect_top_of_book.KRAKEN_TAKER_BPS)
#                   kraken.com/features/fee-schedule
#   OKX perpetual   Regular / Lv1 (<$100k assets, <$5M 30d volume), USDT-settled:
#                   maker 0.020% -> 2.0 bps (taker 0.050%).  okx.com/fees
#                   NOTE: USDC-settled contracts are 0.018% maker; the collection
#                   is USDT-settled (`<BASE>-USDT-SWAP`), so 0.020% applies.
# Both are rebate-free at this tier: a maker PAYS 2 bps, it is not paid. Tiers
# and promotions move - re-verify before trusting downstream numbers, and sweep
# the parameter rather than hardcoding a conclusion.
KRAKEN_MAKER_BPS = 2.0
OKX_MAKER_BPS = 2.0

VENUES = ("kraken", "okx")

# OKX SWAP `sz` is quoted in CONTRACTS, not base units, so a notional needs the
# contract value. ctVal (base units per contract) from
# GET /api/v5/public/instruments?instType=SWAP, read 2026-10-14:
#     BTC-USDT-SWAP   ctVal 0.01 BTC    NEAR-USDT-SWAP  ctVal 10 NEAR
#     UNI-USDT-SWAP   ctVal 1    UNI    ARB-USDT-SWAP   ctVal 10 ARB
# (ctMult is 1 on all four.) Kraken Futures PF_ `bid_size`/`ask_size` are already
# in base currency, so their multiplier is 1.0.
#
# This affects notional/size metrics ONLY. Spread, volatility, markout and the
# edge ratio are all price-derived and unaffected by it.
OKX_CONTRACT_SIZE: dict[str, float] = {"BTC": 0.01, "NEAR": 10.0, "UNI": 1.0, "ARB": 10.0}

# Kraken's `ticker` feed cadence. Any Kraken statistic at or below this horizon
# is at the resolution floor, not a measurement of the market.
KRAKEN_SNAPSHOT_FLOOR_SEC = 1.0

SECONDS_PER_DAY = 86_400


# ============================================================
# Config
# ============================================================

@dataclass
class SpreadFeasibilityConfig:
    """Parameters for the passive spread-capture feasibility analysis.

    Attributes:
        maker_fee_bps_per_side: maker fee in bps per side, per venue. A maker
            PAYS this at the base tier (no rebate).
        horizons_s: markout/volatility horizons in seconds.
        mask_windows: (start, end) UTC strings excluded from all metrics
            (collector/feed outages). Defaults to the Phase B outage mask.
        grid_sec: resampling grid spacing in seconds.
        grid_staleness_sec: max age of a quote when stamping a grid point.
            Older => the grid point is NaN, never forward-filled. Must exceed
            Kraken's ~1 s cadence plus normal reconnect gaps (<=5.6 s observed)
            or Kraken's grid becomes mostly holes.
        inventory_horizon_sec: the "repricing cycle" for the inventory drift
            proxy (5 min).
        tight_spread_multiple: a spread at/below this multiple of the maker
            ROUND-TRIP fee (2 x per-side) is counted as "too tight to make".
        contract_size: base units per quoted size unit, per venue per asset.
            OKX quotes SWAP size in contracts; Kraken quotes base currency.
            A missing entry defaults to 1.0.
        chunksize: rows per read_csv chunk in the streaming loader.
    """

    maker_fee_bps_per_side: dict[str, float] = field(
        default_factory=lambda: {"kraken": KRAKEN_MAKER_BPS, "okx": OKX_MAKER_BPS})
    horizons_s: tuple[int, ...] = (1, 5, 30)
    mask_windows: Sequence[tuple[str, str]] = tuple(PHASE_B_MASK)
    grid_sec: int = 1
    grid_staleness_sec: float = 5.0
    inventory_horizon_sec: int = 300
    tight_spread_multiple: float = 2.0
    contract_size: dict[str, dict[str, float]] = field(
        default_factory=lambda: {"kraken": {}, "okx": dict(OKX_CONTRACT_SIZE)})
    chunksize: int = 1_000_000

    def size_multiplier(self, venue: str, asset: str) -> float:
        """Base units per quoted size unit for (venue, asset); 1.0 if unmapped."""
        return self.contract_size.get(venue, {}).get(asset.upper(), 1.0)

    def fee(self, venue: str) -> float:
        """Maker fee per side, bps, for `venue`."""
        return self.maker_fee_bps_per_side[venue]

    def round_trip_fee(self, venue: str) -> float:
        """Maker round-trip fee, bps: buy at bid + sell at ask, both maker."""
        return 2.0 * self.fee(venue)

    def stream_config(self) -> StreamConfig:
        """The streaming-loader config implied by this config (reuses Phase B masking)."""
        return StreamConfig(
            chunksize=self.chunksize,
            mask_windows=[(pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"))
                          for a, b in self.mask_windows],
        )

    def at_resolution_floor(self, venue: str, horizon_s: float) -> bool:
        """True if `venue`'s feed cannot resolve `horizon_s` (Kraken's 1 s snapshot)."""
        return venue == "kraken" and horizon_s <= KRAKEN_SNAPSHOT_FLOOR_SEC

    def mask_bounds(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Mask windows as UTC Timestamps, accepting either strings or Timestamps."""
        return [(pd.Timestamp(a, tz="UTC") if not isinstance(a, pd.Timestamp) else a,
                 pd.Timestamp(b, tz="UTC") if not isinstance(b, pd.Timestamp) else b)
                for a, b in self.mask_windows]


# ============================================================
# 1 s grid
# ============================================================

def build_second_grid(
    quotes: pd.DataFrame, day: pd.Timestamp, config: SpreadFeasibilityConfig,
) -> pd.DataFrame:
    """Resample one venue-day's quotes onto a regular grid covering the whole UTC day.

    Every day contributes exactly `86400 / grid_sec` rows, so concatenating days
    tiles into one continuous series and horizon windows cross midnight correctly.
    Grid points with no quote within `grid_staleness_sec` are NaN - a stale quote
    is never carried forward, so outage spans become holes rather than fabricated
    flat markets.

    Mask windows are applied TWICE, deliberately: `load_asset_day` drops the raw
    rows, and every grid point inside a window is then nulled here. The second
    pass matters because the as-of tolerance would otherwise let the last quote
    BEFORE a window cover its first `grid_staleness_sec` seconds.

    Args:
        quotes: one venue's quotes, DatetimeIndex (UTC), columns `QUOTE_COLS`.
        day: UTC midnight of the day to cover.
        config: supplies grid spacing and staleness tolerance.

    Returns:
        DataFrame indexed by the grid with columns [bid, bid_size, ask, ask_size, mid].
    """
    day = pd.Timestamp(day)
    if day.tz is None:
        day = day.tz_localize("UTC")
    n = SECONDS_PER_DAY // config.grid_sec
    grid = pd.date_range(day, periods=n, freq=f"{config.grid_sec}s", tz="UTC")
    m = pd.DataFrame(index=grid, columns=[*QUOTE_COLS, "mid"], dtype=float)

    if not quotes.empty:
        q = quotes[QUOTE_COLS].sort_index()
        m = pd.merge_asof(
            pd.DataFrame(index=grid), q,
            left_index=True, right_index=True, direction="backward",
            tolerance=pd.Timedelta(seconds=config.grid_staleness_sec),
        )
        m["mid"] = (m["bid"] + m["ask"]) / 2.0
        # A non-positive or crossed quote is unusable for bps math; drop the point.
        bad = (m["bid"] <= 0) | (m["ask"] <= 0) | (m["ask"] < m["bid"])
        m.loc[bad.fillna(False), :] = np.nan

    for w0, w1 in config.mask_bounds():
        m.loc[(m.index >= w0) & (m.index < w1), :] = np.nan
    return m[[*QUOTE_COLS, "mid"]]


def spread_bps(grid: pd.DataFrame) -> pd.Series:
    """(ask - bid) / mid in bps, on the grid."""
    return 10_000.0 * (grid["ask"] - grid["bid"]) / grid["mid"]


# ============================================================
# Spread / income
# ============================================================

def spread_stats(
    grid: pd.DataFrame, venue: str, config: SpreadFeasibilityConfig,
) -> dict[str, float]:
    """Spread distribution, net half-spread, and the too-tight-to-make time fraction.

    `net_half_spread_bps` = median spread / 2 - maker fee per side: the gross
    income from ONE one-sided fill at the touch, before any adverse selection.

    Two tightness fractions are reported because "too tight to make" has two
    defensible thresholds:
        frac_time_below_round_trip: spread <= maker round trip (break-even).
        frac_time_tight: spread <= `tight_spread_multiple` x round trip (the
            spec's threshold, a 2x margin over break-even by default).
    """
    s = spread_bps(grid).to_numpy(dtype=float)
    valid = np.isfinite(s)
    n = int(valid.sum())
    if n == 0:
        return {"n_grid": 0}
    sv = s[valid]
    fee = config.fee(venue)
    rt = config.round_trip_fee(venue)
    median = float(np.median(sv))
    return {
        "n_grid": n,
        "median_spread_bps": median,
        "p10_spread_bps": float(np.percentile(sv, 10)),
        "p90_spread_bps": float(np.percentile(sv, 90)),
        "mean_spread_bps": float(np.mean(sv)),
        "maker_fee_bps_per_side": fee,
        "maker_round_trip_bps": rt,
        "net_half_spread_bps": median / 2.0 - fee,
        "frac_time_below_round_trip": float(np.mean(sv <= rt)),
        "frac_time_tight": float(np.mean(sv <= config.tight_spread_multiple * rt)),
        "tight_threshold_bps": config.tight_spread_multiple * rt,
    }


def queue_stats(grid: pd.DataFrame, size_multiplier: float = 1.0) -> dict[str, float]:
    """USD size resting at the touch - what a newly posted quote queues behind.

    Bid-side and ask-side touch notionals are pooled; a new quote joins one side,
    and the two are statistically interchangeable at this resolution.

    Args:
        grid: the 1 s grid for one asset x venue.
        size_multiplier: base units per quoted size unit (OKX quotes SWAP size in
            CONTRACTS, so this is ctVal; Kraken quotes base currency, so 1.0).
            Getting this wrong misstates every notional by the contract value -
            OKX BTC is 100x out without it.
    """
    mid = grid["mid"].to_numpy(dtype=float)
    usd = np.concatenate([
        grid["bid_size"].to_numpy(dtype=float) * size_multiplier * mid,
        grid["ask_size"].to_numpy(dtype=float) * size_multiplier * mid,
    ])
    usd = usd[np.isfinite(usd)]
    if not len(usd):
        return {"n": 0, "size_multiplier": size_multiplier}
    return {
        "n": int(len(usd)),
        "size_multiplier": size_multiplier,
        "median_touch_usd": float(np.median(usd)),
        "p90_touch_usd": float(np.percentile(usd, 90)),
        "mean_touch_usd": float(np.mean(usd)),
    }


def inventory_drift_stats(
    grid: pd.DataFrame, config: SpreadFeasibilityConfig,
) -> dict[str, float]:
    """Mid-return distribution over the repricing cycle (default 5 min), in bps.

    This is the cost of carrying one-sided inventory between requotes: a maker
    left long after an adverse fill wears this distribution until it can reprice.
    """
    steps = int(config.inventory_horizon_sec // config.grid_sec)
    mid = grid["mid"]
    r = (10_000.0 * np.log(mid.shift(-steps) / mid)).to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if not len(r):
        return {"n": 0, "horizon_sec": config.inventory_horizon_sec}
    return {
        "n": int(len(r)),
        "horizon_sec": config.inventory_horizon_sec,
        "std_bps": float(np.std(r)),
        "p10_bps": float(np.percentile(r, 10)),
        "p90_bps": float(np.percentile(r, 90)),
        "mean_abs_bps": float(np.mean(np.abs(r))),
    }


# ============================================================
# Adverse markout
# ============================================================

def markout_frame(grid: pd.DataFrame, horizon_s: int, grid_sec: int = 1) -> pd.DataFrame:
    """Per-grid-point adverse flags and markouts for a naive touch quoter.

    At each grid point t the quoter is assumed to have a resting bid at the
    current best bid and a resting ask at the current best ask. Within
    (t, t+tau]:

        adverse_bid: min(best_ask) <= resting_bid   (market traded DOWN through
                     the bid - a fill is certain, and the mid is now lower)
        adverse_ask: max(best_bid) >= resting_ask   (symmetric, upward)

    Markout is the signed mark-to-mid P&L of the resulting position, in bps:

        markout_bid = (mid_{t+tau} - bid_t) / mid_t   (long at the bid)
        markout_ask = (ask_t - mid_{t+tau}) / mid_t   (short at the ask)

    so an ADVERSE outcome is NEGATIVE. `cost_*` is the same quantity negated,
    i.e. a positive cost, which is what the edge ratio divides by.

    Rows where any input in the window is NaN (outage, mask, stale feed) are
    marked `valid=False` and excluded downstream - never counted as non-adverse.
    """
    tau = int(horizon_s // grid_sec)
    if tau < 1:
        raise ValueError(f"horizon_s={horizon_s} is shorter than grid_sec={grid_sec}")

    bid, ask, mid = grid["bid"], grid["ask"], grid["mid"]
    # rolling(tau).min() at j covers (j-tau, j]; shifting by -tau puts the window
    # (i, i+tau] at i. min_periods=tau => NaN if any point in the window is NaN.
    fut_min_ask = ask.rolling(tau, min_periods=tau).min().shift(-tau)
    fut_max_bid = bid.rolling(tau, min_periods=tau).max().shift(-tau)
    mid_fwd = mid.shift(-tau)

    valid = (fut_min_ask.notna() & fut_max_bid.notna() & mid_fwd.notna()
             & bid.notna() & ask.notna() & mid.notna())

    markout_bid = 10_000.0 * (mid_fwd - bid) / mid
    markout_ask = 10_000.0 * (ask - mid_fwd) / mid

    out = pd.DataFrame({
        "valid": valid,
        "adverse_bid": valid & (fut_min_ask <= bid),
        "adverse_ask": valid & (fut_max_bid >= ask),
        "markout_bid_bps": markout_bid.where(valid),
        "markout_ask_bps": markout_ask.where(valid),
        "mid_move_bps": (10_000.0 * np.log(mid_fwd / mid)).where(valid),
    }, index=grid.index)
    out["cost_bid_bps"] = -out["markout_bid_bps"]
    out["cost_ask_bps"] = -out["markout_ask_bps"]
    return out


def _pooled(mk: pd.DataFrame, col_bid: str, col_ask: str, mask_bid, mask_ask) -> np.ndarray:
    """Concatenate the bid-side and ask-side values selected by their masks."""
    a = mk.loc[mask_bid, col_bid].to_numpy(dtype=float)
    b = mk.loc[mask_ask, col_ask].to_numpy(dtype=float)
    v = np.concatenate([a, b])
    return v[np.isfinite(v)]


def summarize_markout(
    mk: pd.DataFrame, horizon_s: int, venue: str, config: SpreadFeasibilityConfig,
    net_half_spread_bps: float, row_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Aggregate an ALREADY-COMPUTED markout frame, optionally over a row subset.

    `mk` must have been built by `markout_frame` on a CONTIGUOUS grid - the
    rolling windows inside it assume adjacent rows are `grid_sec` apart. To get
    per-hour or weekday/weekend figures, build `mk` once on the full grid and
    pass a boolean `row_mask` here; never group the grid first and re-run
    `markout_frame` on the slice, which would splice non-adjacent days together
    and read a multi-hour mid move as a one-second markout.

    `adverse_per_hour` counts SIDE-events per valid hour of grid time. It is a
    per-second-sampled hazard, not a count of distinct fills: consecutive grid
    points re-evaluate an overlapping window, so the same market move is counted
    at several adjacent points. Use `frac_adverse` (the probability that a touch
    quote posted at a random instant is run over within tau) for a
    sampling-independent reading.

    `uncond_markout_*_bps` is the mean cost over ALL valid grid points, not only
    adverse ones. Note the pooled two-sided version is identically
    -(half-spread): a touch quote earns the half-spread whenever the mid does not
    move, so per-side values are reported instead and
    `uncond_abs_mid_move_bps` carries the magnitude of the typical mid excursion.
    """
    if row_mask is not None:
        mk = mk.loc[row_mask]

    valid = mk["valid"].to_numpy(dtype=bool)
    n_valid = int(valid.sum())
    floor = config.at_resolution_floor(venue, horizon_s)

    if n_valid == 0:
        return {"horizon_s": horizon_s, "n_valid": 0, "at_resolution_floor": floor}

    valid_hours = n_valid * config.grid_sec / 3600.0
    ab = mk["adverse_bid"].to_numpy(dtype=bool)
    aa = mk["adverse_ask"].to_numpy(dtype=bool)
    n_adverse = int(ab.sum() + aa.sum())

    adverse_cost = _pooled(mk, "cost_bid_bps", "cost_ask_bps", ab, aa)
    move = mk["mid_move_bps"].to_numpy(dtype=float)
    move = move[np.isfinite(move)]

    mean_adverse_cost = float(np.mean(adverse_cost)) if len(adverse_cost) else float("nan")
    # Edge ratio is only meaningful against a positive cost; a non-positive mean
    # adverse cost means the "adverse" events were not adverse on average.
    if np.isfinite(mean_adverse_cost) and mean_adverse_cost > 0:
        edge_ratio = net_half_spread_bps / mean_adverse_cost
    else:
        edge_ratio = float("nan")

    return {
        "horizon_s": horizon_s,
        "at_resolution_floor": floor,
        "n_valid": n_valid,
        "valid_hours": valid_hours,
        "sigma_mid_bps": float(np.std(move)) if len(move) else float("nan"),
        "n_adverse_events": n_adverse,
        "adverse_per_hour": n_adverse / valid_hours if valid_hours > 0 else float("nan"),
        "frac_adverse": n_adverse / (2.0 * n_valid),
        "mean_adverse_cost_bps": mean_adverse_cost,
        "median_adverse_cost_bps": float(np.median(adverse_cost)) if len(adverse_cost)
                                   else float("nan"),
        "uncond_markout_bid_bps": float(np.nanmean(mk["cost_bid_bps"].to_numpy(dtype=float))),
        "uncond_markout_ask_bps": float(np.nanmean(mk["cost_ask_bps"].to_numpy(dtype=float))),
        "uncond_abs_mid_move_bps": float(np.mean(np.abs(move))) if len(move) else float("nan"),
        "net_half_spread_bps": net_half_spread_bps,
        "edge_ratio": edge_ratio,
    }


def horizon_stats(
    grid: pd.DataFrame, horizon_s: int, venue: str, config: SpreadFeasibilityConfig,
    net_half_spread_bps: float,
) -> dict[str, Any]:
    """`markout_frame` + `summarize_markout` over a whole contiguous grid."""
    mk = markout_frame(grid, horizon_s, config.grid_sec)
    return summarize_markout(mk, horizon_s, venue, config, net_half_spread_bps)


# ============================================================
# Per-venue analysis
# ============================================================

def analyze_venue(
    grid: pd.DataFrame, venue: str, asset: str, config: SpreadFeasibilityConfig,
) -> dict[str, Any]:
    """Every feasibility statistic for one asset x venue full-collection grid.

    The markout frame for the shortest horizon is built ONCE on the contiguous
    grid and reused for the hour-of-day and weekday/weekend buckets, so no
    bucket ever splices non-adjacent days together.
    """
    sp = spread_stats(grid, venue, config)
    nhs = sp.get("net_half_spread_bps", float("nan"))
    out: dict[str, Any] = {
        "venue": venue,
        "spread": sp,
        "queue": queue_stats(grid, config.size_multiplier(venue, asset)),
        "inventory": inventory_drift_stats(grid, config),
        "horizons": {h: horizon_stats(grid, h, venue, config, nhs)
                     for h in config.horizons_s},
    }
    if sp.get("n_grid"):
        tau = min(config.horizons_s)
        mk = markout_frame(grid, tau, config.grid_sec)
        out["bucket_horizon_s"] = tau
        out["by_hour"] = {
            int(h): _bucket_stats(grid, mk, grid.index.hour == h, venue, config, tau)
            for h in range(24)}
        we = grid.index.weekday >= 5
        out["by_week"] = {
            "weekday": _bucket_stats(grid, mk, ~we, venue, config, tau),
            "weekend": _bucket_stats(grid, mk, we, venue, config, tau),
        }
    return out


def _bucket_stats(
    grid: pd.DataFrame, mk: pd.DataFrame, row_mask: np.ndarray, venue: str,
    config: SpreadFeasibilityConfig, horizon_s: int,
) -> dict[str, float]:
    """Spread + edge-ratio summary for the grid rows selected by `row_mask`.

    `mk` is the full-grid markout frame; only its ROWS are subset, never
    recomputed, so the windows behind it stay anchored to real adjacent seconds.
    """
    sp = spread_stats(grid.loc[row_mask], venue, config)
    if not sp.get("n_grid"):
        return {"n_grid": 0}
    h = summarize_markout(mk, horizon_s, venue, config,
                          sp["net_half_spread_bps"], row_mask)
    return {
        "n_grid": sp["n_grid"],
        "median_spread_bps": sp["median_spread_bps"],
        "net_half_spread_bps": sp["net_half_spread_bps"],
        "frac_time_tight": sp["frac_time_tight"],
        "sigma_mid_bps": h.get("sigma_mid_bps", float("nan")),
        "adverse_per_hour": h.get("adverse_per_hour", float("nan")),
        "mean_adverse_cost_bps": h.get("mean_adverse_cost_bps", float("nan")),
        "edge_ratio": h.get("edge_ratio", float("nan")),
    }


# ============================================================
# Streaming over the collection
# ============================================================

def stream_grid(
    out_dir: Path | str, asset: str, config: SpreadFeasibilityConfig,
) -> dict[str, pd.DataFrame]:
    """Build each venue's full-collection 1 s grid, one daily file at a time.

    The raw CSVs are 71M rows, so they are read in chunks and only one asset's
    rows are retained per file (see `streaming.load_asset_day`); the per-day
    grid is a few tens of thousands of rows, so the retained result for the
    whole collection is ~1M rows per venue regardless of the input size.
    """
    sc = config.stream_config()
    per_venue: dict[str, list[pd.DataFrame]] = {v: [] for v in VENUES}
    for path in daily_files(out_dir):
        day = pd.Timestamp(path.stem.replace("tob_", ""), tz="UTC")
        k, o = load_asset_day(path, asset, sc)
        for venue, q in (("kraken", k), ("okx", o)):
            per_venue[venue].append(build_second_grid(q, day, config))
        logger.info("%s %s: kraken=%d okx=%d quote rows", asset, day.date(), len(k), len(o))
        del k, o
    return {v: pd.concat(parts).sort_index() if parts else pd.DataFrame()
            for v, parts in per_venue.items()}


def analyze_asset(
    out_dir: Path | str, asset: str, config: SpreadFeasibilityConfig,
) -> dict[str, Any]:
    """Full feasibility result for one asset, both venues."""
    grids = stream_grid(out_dir, asset, config)
    return {
        "asset": asset,
        "venues": {v: analyze_venue(g, v, asset, config)
                   for v, g in grids.items() if not g.empty},
    }


# ============================================================
# Formatting
# ============================================================

def format_summary(results: dict[str, dict[str, Any]], config: SpreadFeasibilityConfig) -> str:
    """Cross-asset summary: one row per asset x venue. BTC rows are the control."""
    taus = list(config.horizons_s)
    t0, t1 = taus[0], (taus[1] if len(taus) > 1 else taus[0])
    L = ["=" * 118]
    L.append("  PASSIVE SPREAD-CAPTURE FEASIBILITY (Path B) - per-fill economics, quotes only")
    fees = ", ".join(f"{v} {config.fee(v):.1f}" for v in VENUES)
    L.append(f"  maker fee bps/side: {fees}  (round trip {config.round_trip_fee('okx'):.1f} bps)"
             f"   grid {config.grid_sec}s, staleness tol {config.grid_staleness_sec:g}s")
    L.append("  NO fill rates are derivable from quotes: these are per-fill ratios, not daily dollars.")
    L.append("  '!' = at the Kraken ~1s snapshot floor (understates vol and adverse selection).")
    L.append("  touch $ applies OKX ctVal (BTC .01 / NEAR 10 / UNI 1 / ARB 10 base units per "
             "contract); Kraken is base units.")
    L.append("=" * 118)
    L.append("")
    L.append(f"{'asset':<6}{'venue':<8}{'med spr':>9}{'net half':>9}"
             f"{'sig' + str(t0) + 's':>8}{'sig' + str(t1) + 's':>8}"
             f"{'adv/hr':>9}{'advcost' + str(t0) + 's':>10}{'advcost' + str(t1) + 's':>10}"
             f"{'EDGE' + str(t0) + 's':>9}{'EDGE' + str(t1) + 's':>9}"
             f"{'touch $':>10}{'tight%':>8}")
    for asset, res in results.items():
        for venue in VENUES:
            v = res["venues"].get(venue)
            if not v:
                continue
            sp, q = v["spread"], v["queue"]
            h0, h1 = v["horizons"].get(t0, {}), v["horizons"].get(t1, {})
            flag = "!" if config.at_resolution_floor(venue, t0) else " "
            L.append(
                f"{asset:<6}{venue + flag:<8}"
                f"{sp['median_spread_bps']:>9.2f}{sp['net_half_spread_bps']:>9.2f}"
                f"{h0.get('sigma_mid_bps', float('nan')):>8.2f}"
                f"{h1.get('sigma_mid_bps', float('nan')):>8.2f}"
                f"{h0.get('adverse_per_hour', float('nan')):>9.1f}"
                f"{h0.get('mean_adverse_cost_bps', float('nan')):>10.2f}"
                f"{h1.get('mean_adverse_cost_bps', float('nan')):>10.2f}"
                f"{h0.get('edge_ratio', float('nan')):>9.2f}"
                f"{h1.get('edge_ratio', float('nan')):>9.2f}"
                f"{q.get('median_touch_usd', float('nan')):>10,.0f}"
                f"{sp['frac_time_tight'] * 100:>8.1f}")
    return "\n".join(L)


def format_detail(asset: str, res: dict[str, Any], config: SpreadFeasibilityConfig) -> str:
    """Per-horizon, spread-distribution, queue and inventory detail for one asset."""
    L = ["", "-" * 118, f"  {asset}", "-" * 118]
    for venue in VENUES:
        v = res["venues"].get(venue)
        if not v:
            continue
        sp, q, inv = v["spread"], v["queue"], v["inventory"]
        L.append(f"  {venue}: {sp['n_grid']:,} valid grid points "
                 f"({sp['n_grid'] * config.grid_sec / 3600:.1f} h)")
        L.append(f"    spread bps  p10 {sp['p10_spread_bps']:.2f}  "
                 f"median {sp['median_spread_bps']:.2f}  p90 {sp['p90_spread_bps']:.2f}  "
                 f"mean {sp['mean_spread_bps']:.2f}")
        L.append(f"    net half-spread {sp['net_half_spread_bps']:+.2f} bps "
                 f"(= median/2 - {sp['maker_fee_bps_per_side']:.1f} fee)   "
                 f"time spread <= round-trip fee ({sp['maker_round_trip_bps']:.1f}): "
                 f"{sp['frac_time_below_round_trip'] * 100:.1f}%   "
                 f"<= {sp['tight_threshold_bps']:.1f}: {sp['frac_time_tight'] * 100:.1f}%")
        L.append(f"    touch size USD  median {q['median_touch_usd']:,.0f}  "
                 f"p90 {q['p90_touch_usd']:,.0f}   "
                 f"(size x {q['size_multiplier']:g} base units/unit x mid)")
        L.append(f"    {inv['horizon_sec']}s mid return bps  std {inv['std_bps']:.1f}  "
                 f"p10 {inv['p10_bps']:.1f}  p90 {inv['p90_bps']:.1f}  "
                 f"mean|r| {inv['mean_abs_bps']:.1f}")
        L.append(f"      {'tau':>5}{'sigma':>9}{'adv/hr':>10}{'P(adv)':>9}"
                 f"{'advcost':>10}{'med cost':>10}{'|dmid|':>9}{'EDGE':>8}  floor")
        for h in config.horizons_s:
            s = v["horizons"].get(h, {})
            if not s.get("n_valid"):
                continue
            L.append(f"      {h:>5}{s['sigma_mid_bps']:>9.2f}{s['adverse_per_hour']:>10.1f}"
                     f"{s['frac_adverse'] * 100:>8.2f}%{s['mean_adverse_cost_bps']:>10.2f}"
                     f"{s['median_adverse_cost_bps']:>10.2f}"
                     f"{s['uncond_abs_mid_move_bps']:>9.2f}{s['edge_ratio']:>8.2f}"
                     f"  {'YES' if s['at_resolution_floor'] else '-'}")
        wk = v.get("by_week", {})
        if wk:
            for label in ("weekday", "weekend"):
                b = wk.get(label, {})
                if not b.get("n_grid"):
                    continue
                L.append(f"    {label:<8} med spr {b['median_spread_bps']:.2f}  "
                         f"net half {b['net_half_spread_bps']:+.2f}  "
                         f"adv/hr {b['adverse_per_hour']:.1f}  "
                         f"advcost {b['mean_adverse_cost_bps']:.2f}  "
                         f"EDGE {b['edge_ratio']:.2f}")
    return "\n".join(L)


def format_hour_table(
    asset: str, res: dict[str, Any], config: SpreadFeasibilityConfig,
) -> str:
    """Hour-of-day (UTC) x tau=min(horizons) table for one asset."""
    tau = min(config.horizons_s)
    L = ["", f"### {asset} - hour-of-day (UTC), tau = {tau}s", ""]
    L.append(f"{'hr':<4}" + "".join(
        f"{v[:3] + ' spr':>9}{v[:3] + ' net':>9}{v[:3] + ' adv/h':>11}"
        f"{v[:3] + ' cost':>10}{v[:3] + ' EDGE':>10}"
        for v in VENUES if v in res["venues"]))
    for h in range(24):
        line = f"{h:02d}  "
        for venue in VENUES:
            v = res["venues"].get(venue)
            if not v:
                continue
            b = v.get("by_hour", {}).get(h, {})
            if not b.get("n_grid"):
                line += f"{'-':>9}{'-':>9}{'-':>11}{'-':>10}{'-':>10}"
                continue
            line += (f"{b['median_spread_bps']:>9.2f}{b['net_half_spread_bps']:>9.2f}"
                     f"{b['adverse_per_hour']:>11.1f}{b['mean_adverse_cost_bps']:>10.2f}"
                     f"{b['edge_ratio']:>10.2f}")
        L.append(line)
    return "\n".join(L)


# ============================================================
# CLI
# ============================================================

def _json_default(o: Any) -> Any:
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Passive spread-capture feasibility from the Phase B quote set.")
    parser.add_argument("--dir", default="data/tob")
    parser.add_argument("--assets", default="NEAR,UNI,ARB,BTC",
                        help="BTC is the control.")
    parser.add_argument("--kraken-maker-bps", type=float, default=KRAKEN_MAKER_BPS)
    parser.add_argument("--okx-maker-bps", type=float, default=OKX_MAKER_BPS)
    parser.add_argument("--horizons", default="1,5,30", help="Markout horizons, seconds.")
    parser.add_argument("--grid-sec", type=int, default=1)
    parser.add_argument("--staleness-sec", type=float, default=5.0)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--hour-table", default="NEAR,UNI",
                        help="Assets to print the hour-of-day table for.")
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = SpreadFeasibilityConfig(
        maker_fee_bps_per_side={"kraken": args.kraken_maker_bps, "okx": args.okx_maker_bps},
        horizons_s=tuple(int(h) for h in args.horizons.split(",")),
        mask_windows=() if args.no_mask else tuple(PHASE_B_MASK),
        grid_sec=args.grid_sec,
        grid_staleness_sec=args.staleness_sec,
        chunksize=args.chunksize,
    )

    assets = [a.strip().upper() for a in args.assets.split(",")]
    results = {a: analyze_asset(args.dir, a, config) for a in assets}

    print(format_summary(results, config))
    for asset in assets:
        print(format_detail(asset, results[asset], config))
    for asset in [a.strip().upper() for a in args.hour_table.split(",") if a.strip()]:
        if asset in results:
            print(format_hour_table(asset, results[asset], config))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, default=_json_default)
        logger.info("Wrote %s", args.json_out)


if __name__ == "__main__":
    main()
