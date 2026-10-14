"""
tests/test_spread_feasibility.py - Unit tests for
src/strategies/mm/spread_feasibility.py

Hand-built quotes and synthetic daily CSVs in tmp_path; no network. Guards the
arithmetic the feasibility headline rests on: spread and net-half-spread math,
1 s grid resampling with a staleness tolerance (never forward-filled across a
hole), adverse-event detection on a series where the ask is driven through a
resting bid at known times, the markout SIGN convention (adverse is negative for
a bid), the edge ratio on a constructed case, mask application, and that the
streaming per-day build equals a single-frame build on a multi-day fixture.

Run:
    pytest tests/test_spread_feasibility.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.collect_top_of_book import TOB_COLUMNS  # noqa: E402
from src.strategies.mm import spread_feasibility as sf  # noqa: E402

DAY = pd.Timestamp("2026-10-05", tz="UTC")


# ============================================================
# Builders
# ============================================================

def quotes(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    """Build a venue quote frame from (offset_sec, bid, bid_size, ask, ask_size)."""
    idx = pd.DatetimeIndex([DAY + pd.Timedelta(seconds=r[0]) for r in rows], name="ts_recv")
    return pd.DataFrame(
        {"bid": [r[1] for r in rows], "bid_size": [r[2] for r in rows],
         "ask": [r[3] for r in rows], "ask_size": [r[4] for r in rows]},
        index=idx)


def flat_quotes(n: int, bid: float = 99.99, ask: float = 100.01,
                size: float = 10.0, step: float = 1.0) -> pd.DataFrame:
    return quotes([(i * step, bid, size, ask, size) for i in range(n)])


def cfg(**kw) -> sf.SpreadFeasibilityConfig:
    kw.setdefault("mask_windows", ())
    kw.setdefault("maker_fee_bps_per_side", {"kraken": 2.0, "okx": 2.0})
    return sf.SpreadFeasibilityConfig(**kw)


def write_day(out_dir: Path, day: str, rows: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tob_{day}.csv"
    pd.DataFrame(rows, columns=TOB_COLUMNS).to_csv(path, index=False)
    return path


def csv_rows(day: str, asset: str, n: int, bid: float = 99.99, ask: float = 100.01,
             venues=("kraken", "okx")) -> list[dict]:
    t0 = int(pd.Timestamp(day, tz="UTC").value // 1_000_000)
    out = []
    for i in range(n):
        for v in venues:
            out.append({"ts_recv": t0 + i * 1000, "ts_exchange": t0 + i * 1000 - 20,
                        "venue": v, "asset": asset, "bid": bid, "bid_size": 10.0,
                        "ask": ask, "ask_size": 10.0})
    return out


# ============================================================
# Fees
# ============================================================

def test_round_trip_is_twice_the_per_side_fee():
    c = cfg()
    assert c.fee("okx") == 2.0
    assert c.round_trip_fee("okx") == 4.0


def test_kraken_resolution_floor_flag():
    c = cfg()
    assert c.at_resolution_floor("kraken", 1) is True
    assert c.at_resolution_floor("kraken", 5) is False
    assert c.at_resolution_floor("okx", 1) is False   # bbo-tbt is tick-by-tick


# ============================================================
# Grid resampling
# ============================================================

def test_grid_covers_the_whole_day_at_one_second():
    g = sf.build_second_grid(flat_quotes(10), DAY, cfg())

    assert len(g) == 86_400
    assert g.index[0] == DAY
    assert g.index[1] - g.index[0] == pd.Timedelta(seconds=1)
    assert list(g.columns) == ["bid", "bid_size", "ask", "ask_size", "mid"]


def test_grid_stamps_the_last_quote_and_computes_mid():
    g = sf.build_second_grid(flat_quotes(5), DAY, cfg())

    assert g["bid"].iloc[0] == pytest.approx(99.99)
    assert g["mid"].iloc[0] == pytest.approx(100.0)
    # Inside the tolerance (5 s default) the last quote is carried.
    assert np.isfinite(g["mid"].iloc[7])


def test_grid_does_not_forward_fill_past_the_staleness_tolerance():
    """A hole must stay a hole: a stale quote is exactly what fakes a quiet market."""
    g = sf.build_second_grid(flat_quotes(3), DAY, cfg(grid_staleness_sec=2.0))

    assert np.isfinite(g["mid"].iloc[2])      # 2 s after the last quote: inside
    assert not np.isfinite(g["mid"].iloc[9])  # 7 s after: dropped
    assert g["mid"].notna().sum() == 5        # quotes at t=0,1,2 each cover 2 s


def test_grid_before_first_quote_is_nan():
    g = sf.build_second_grid(quotes([(100.0, 99.99, 10.0, 100.01, 10.0)]), DAY, cfg())

    assert not np.isfinite(g["mid"].iloc[0])
    assert np.isfinite(g["mid"].iloc[100])


def test_grid_with_no_quotes_is_all_nan_but_full_length():
    """A venue silent for a day must still tile the grid, or horizons misalign."""
    empty = flat_quotes(0)

    g = sf.build_second_grid(empty, DAY, cfg())

    assert len(g) == 86_400 and g["mid"].notna().sum() == 0


def test_grid_drops_crossed_and_nonpositive_quotes():
    q = quotes([(0, 100.0, 10.0, 99.0, 10.0),    # crossed
                (10, 0.0, 10.0, 100.01, 10.0),   # non-positive bid
                (20, 99.99, 10.0, 100.01, 10.0)])  # good

    g = sf.build_second_grid(q, DAY, cfg(grid_staleness_sec=1.0))

    assert not np.isfinite(g["mid"].iloc[0])
    assert not np.isfinite(g["mid"].iloc[10])
    assert np.isfinite(g["mid"].iloc[20])


# ============================================================
# Spread / income math
# ============================================================

def test_spread_bps_and_net_half_spread():
    # bid 99.90 / ask 100.10 -> mid 100.0, spread 0.20 -> 20 bps.
    g = sf.build_second_grid(flat_quotes(20, bid=99.90, ask=100.10), DAY, cfg())

    s = sf.spread_stats(g, "okx", cfg())

    assert s["median_spread_bps"] == pytest.approx(20.0)
    # net half-spread = 20/2 - 2 = 8 bps per one-sided fill.
    assert s["net_half_spread_bps"] == pytest.approx(8.0)
    assert s["maker_round_trip_bps"] == pytest.approx(4.0)


def test_net_half_spread_is_negative_when_the_spread_cannot_cover_the_fee():
    # 2 bps spread -> half is 1 bps, under the 2 bps maker fee.
    g = sf.build_second_grid(flat_quotes(20, bid=99.99, ask=100.01), DAY, cfg())

    s = sf.spread_stats(g, "okx", cfg())

    assert s["median_spread_bps"] == pytest.approx(2.0)
    assert s["net_half_spread_bps"] == pytest.approx(-1.0)


def test_tight_fractions_use_both_thresholds():
    """Break-even is spread <= round trip; the spec's 'tight' adds a 2x margin."""
    g = sf.build_second_grid(flat_quotes(30, bid=99.97, ask=100.03), DAY, cfg())
    # spread = 0.06/100 = 6 bps. Round trip 4, tight threshold 8.

    s = sf.spread_stats(g, "okx", cfg())

    assert s["median_spread_bps"] == pytest.approx(6.0)
    assert s["frac_time_below_round_trip"] == pytest.approx(0.0)   # 6 > 4
    assert s["frac_time_tight"] == pytest.approx(1.0)              # 6 <= 8


def test_queue_stats_pool_both_sides_in_usd():
    g = sf.build_second_grid(flat_quotes(10, size=3.0), DAY, cfg())

    q = sf.queue_stats(g)

    assert q["median_touch_usd"] == pytest.approx(3.0 * 100.0, rel=1e-3)
    assert q["size_multiplier"] == 1.0


# ============================================================
# Contract size (OKX quotes SWAP size in contracts, not base units)
# ============================================================

def test_queue_stats_applies_the_contract_multiplier():
    """OKX BTC is 100x out without ctVal: 0.01 BTC per contract."""
    g = sf.build_second_grid(flat_quotes(10, size=600.0), DAY, cfg())

    raw = sf.queue_stats(g, size_multiplier=1.0)
    scaled = sf.queue_stats(g, size_multiplier=0.01)

    assert raw["median_touch_usd"] == pytest.approx(600.0 * 100.0, rel=1e-3)
    assert scaled["median_touch_usd"] == pytest.approx(600.0 * 0.01 * 100.0, rel=1e-3)
    assert scaled["size_multiplier"] == 0.01


@pytest.mark.parametrize("asset,expected", [
    ("BTC", 0.01), ("NEAR", 10.0), ("UNI", 1.0), ("ARB", 10.0),
])
def test_okx_contract_sizes_match_the_published_ctval(asset, expected):
    """ctVal from GET /api/v5/public/instruments?instType=SWAP, read 2026-10-14."""
    c = cfg()
    assert c.size_multiplier("okx", asset) == expected
    assert c.size_multiplier("kraken", asset) == 1.0   # Kraken PF_ is base units


def test_size_multiplier_defaults_to_one_for_an_unmapped_asset():
    assert cfg().size_multiplier("okx", "DOGE") == 1.0


def test_contract_multiplier_does_not_touch_price_metrics():
    """The headline (spread, sigma, markout, edge ratio) must be size-independent."""
    # A permanent step down at t=30 so there IS a finite edge ratio to compare.
    rows = [(i, 99.90, 10.0, 100.10, 10.0) if i < 30 else (i, 99.70, 10.0, 99.90, 10.0)
            for i in range(60)]
    g = sf.build_second_grid(quotes(rows), DAY, cfg(grid_staleness_sec=1.0))
    c_a = cfg(grid_staleness_sec=1.0, contract_size={"kraken": {}, "okx": {"NEAR": 1.0}})
    c_b = cfg(grid_staleness_sec=1.0, contract_size={"kraken": {}, "okx": {"NEAR": 10.0}})

    a = sf.analyze_venue(g, "okx", "NEAR", c_a)
    b = sf.analyze_venue(g, "okx", "NEAR", c_b)

    assert a["spread"] == b["spread"]
    assert a["horizons"][1]["sigma_mid_bps"] == b["horizons"][1]["sigma_mid_bps"]
    assert np.isfinite(a["horizons"][1]["edge_ratio"])
    assert a["horizons"][1]["edge_ratio"] == b["horizons"][1]["edge_ratio"]
    # Only the notional moves, by exactly the multiplier ratio.
    assert b["queue"]["median_touch_usd"] == pytest.approx(
        10.0 * a["queue"]["median_touch_usd"], rel=1e-6)


# ============================================================
# Adverse-event detection
# ============================================================

def _drive_ask_through_bid_at(t_adverse: int, n: int = 60) -> pd.DataFrame:
    """Flat 99.99/100.01 book, except the whole book steps DOWN at `t_adverse`
    so the best ask (99.95) is below the bid resting at 99.99."""
    rows = []
    for i in range(n):
        if i == t_adverse:
            rows.append((i, 99.93, 10.0, 99.95, 10.0))
        else:
            rows.append((i, 99.99, 10.0, 100.01, 10.0))
    return quotes(rows)


def test_adverse_bid_detected_exactly_when_the_ask_trades_through():
    g = sf.build_second_grid(_drive_ask_through_bid_at(10), DAY, cfg(grid_staleness_sec=1.0))

    mk = sf.markout_frame(g, horizon_s=1, grid_sec=1)

    # At t=9 the next second's ask (99.95) is <= the bid resting at 99.99.
    assert bool(mk["adverse_bid"].iloc[9])
    assert not bool(mk["adverse_bid"].iloc[5])     # quiet stretch
    assert not bool(mk["adverse_ask"].iloc[9])     # the move was downward only


def test_adverse_window_is_exclusive_of_t_and_inclusive_of_t_plus_tau():
    g = sf.build_second_grid(_drive_ask_through_bid_at(10), DAY, cfg(grid_staleness_sec=1.0))

    mk5 = sf.markout_frame(g, horizon_s=5, grid_sec=1)

    # tau=5 at t: window (t, t+5]. t=5..9 see the t=10 break; t=10 does not
    # (its own point is excluded), and t=4 is too early.
    assert [bool(mk5["adverse_bid"].iloc[i]) for i in range(4, 12)] == \
           [False, True, True, True, True, True, False, False]


def test_adverse_ask_detected_on_an_upward_trade_through():
    rows = [(i, 99.99, 10.0, 100.01, 10.0) for i in range(30)]
    rows[20] = (20, 100.05, 10.0, 100.07, 10.0)   # book steps UP through the ask

    g = sf.build_second_grid(quotes(rows), DAY, cfg(grid_staleness_sec=1.0))
    mk = sf.markout_frame(g, horizon_s=1, grid_sec=1)

    assert bool(mk["adverse_ask"].iloc[19])
    assert not bool(mk["adverse_bid"].iloc[19])


def test_markout_sign_adverse_is_negative_for_a_bid():
    """The sign convention: a bid run over by a falling market marks NEGATIVE."""
    g = sf.build_second_grid(_drive_ask_through_bid_at(10), DAY, cfg(grid_staleness_sec=1.0))

    mk = sf.markout_frame(g, horizon_s=1, grid_sec=1)

    # Bought at 99.99; one second later mid = 99.94 -> markout < 0, cost > 0.
    assert mk["markout_bid_bps"].iloc[9] < 0
    assert mk["cost_bid_bps"].iloc[9] == pytest.approx(-mk["markout_bid_bps"].iloc[9])
    assert mk["cost_bid_bps"].iloc[9] == pytest.approx(
        10_000.0 * (99.99 - 99.94) / 100.0, rel=1e-3)


def test_markout_is_positive_when_the_market_does_not_move():
    """Unmoved mid: a touch quote earns its half-spread, so markout is positive."""
    g = sf.build_second_grid(flat_quotes(30, bid=99.90, ask=100.10), DAY, cfg())

    mk = sf.markout_frame(g, horizon_s=1, grid_sec=1)

    assert mk["markout_bid_bps"].iloc[5] == pytest.approx(10.0, rel=1e-6)   # half of 20 bps
    assert mk["markout_ask_bps"].iloc[5] == pytest.approx(10.0, rel=1e-6)


def test_invalid_rows_are_excluded_not_counted_as_non_adverse():
    """A hole must not read as 'quiet' - that is the stale-feed trap."""
    g = sf.build_second_grid(flat_quotes(3), DAY, cfg(grid_staleness_sec=1.0))

    mk = sf.markout_frame(g, horizon_s=1, grid_sec=1)

    assert bool(mk["valid"].iloc[0])
    assert not bool(mk["valid"].iloc[50])          # no quote there at all
    assert not bool(mk["adverse_bid"].iloc[50])
    assert mk["valid"].sum() < 10                  # only the covered seconds


def test_markout_frame_rejects_a_horizon_below_the_grid():
    g = sf.build_second_grid(flat_quotes(5), DAY, cfg())

    with pytest.raises(ValueError, match="shorter than"):
        sf.markout_frame(g, horizon_s=1, grid_sec=5)


# ============================================================
# Horizon stats + the edge ratio
# ============================================================

def test_edge_ratio_on_a_constructed_case():
    """20 bps spread, 2 bps fee -> 8 bps income; one 10 bps adverse hit -> ratio 0.8."""
    n = 60
    # Book sits at 99.90/100.10, then steps PERMANENTLY down 0.20 at t=30, putting
    # the new ask (99.90) exactly at the bid resting from t=29 -> one trade-through.
    # A permanent step (not a spike) is used on purpose: a spike would snap back up
    # through the resting ask and register a second, opposite adverse event.
    rows = [(i, 99.90, 10.0, 100.10, 10.0) if i < 30 else (i, 99.70, 10.0, 99.90, 10.0)
            for i in range(n)]

    g = sf.build_second_grid(quotes(rows), DAY, cfg(grid_staleness_sec=1.0))
    sp = sf.spread_stats(g, "okx", cfg())
    h = sf.horizon_stats(g, 1, "okx", cfg(), sp["net_half_spread_bps"])

    # 20 bps before the step, 20.04 after (same 0.20 spread on a lower mid).
    assert sp["net_half_spread_bps"] == pytest.approx(8.0, abs=0.05)
    assert h["n_adverse_events"] == 1                       # only t=29 -> t=30
    # Bought at 99.90; one second later mid = 99.80 -> 10 bps cost.
    assert h["mean_adverse_cost_bps"] == pytest.approx(10.0, rel=1e-2)
    assert h["edge_ratio"] == pytest.approx(0.8, rel=1e-2)


def test_a_round_trip_spike_registers_both_sides():
    """A down-spike that snaps back is adverse on the bid AND then on the ask."""
    rows = [(i, 99.90, 10.0, 100.10, 10.0) for i in range(40)]
    rows[30] = (30, 99.70, 10.0, 99.90, 10.0)   # down through the bid, one tick only

    g = sf.build_second_grid(quotes(rows), DAY, cfg(grid_staleness_sec=1.0))
    mk = sf.markout_frame(g, horizon_s=1, grid_sec=1)

    assert bool(mk["adverse_bid"].iloc[29])     # market fell through the resting bid
    assert bool(mk["adverse_ask"].iloc[30])     # then rose back through the resting ask
    h = sf.horizon_stats(g, 1, "okx", cfg(), 8.0)
    assert h["n_adverse_events"] == 2


def test_edge_ratio_is_nan_without_adverse_events():
    g = sf.build_second_grid(flat_quotes(60, bid=99.90, ask=100.10), DAY, cfg())

    h = sf.horizon_stats(g, 1, "okx", cfg(), 8.0)

    assert h["n_adverse_events"] == 0
    assert np.isnan(h["edge_ratio"])
    assert h["sigma_mid_bps"] == pytest.approx(0.0)


def test_adverse_rate_per_hour_uses_valid_grid_time():
    g = sf.build_second_grid(_drive_ask_through_bid_at(10, n=3601),
                             DAY, cfg(grid_staleness_sec=1.0))

    h = sf.horizon_stats(g, 1, "okx", cfg(), 8.0)

    assert h["valid_hours"] == pytest.approx(1.0, rel=0.01)
    assert h["adverse_per_hour"] == pytest.approx(h["n_adverse_events"], rel=0.02)


def test_sigma_rises_with_horizon_on_a_random_walk():
    rng = np.random.default_rng(0)
    mid = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-4, 3600)))
    q = quotes([(i, m - 0.01, 10.0, m + 0.01, 10.0) for i, m in enumerate(mid)])
    g = sf.build_second_grid(q, DAY, cfg(grid_staleness_sec=1.0))
    c = cfg()

    s1 = sf.horizon_stats(g, 1, "okx", c, 8.0)["sigma_mid_bps"]
    s30 = sf.horizon_stats(g, 30, "okx", c, 8.0)["sigma_mid_bps"]

    assert s30 > s1                                  # sqrt-time scaling
    assert s30 == pytest.approx(s1 * np.sqrt(30), rel=0.35)


def test_inventory_drift_uses_the_configured_horizon():
    rng = np.random.default_rng(1)
    mid = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-4, 2000)))
    q = quotes([(i, m - 0.01, 10.0, m + 0.01, 10.0) for i, m in enumerate(mid)])
    g = sf.build_second_grid(q, DAY, cfg(grid_staleness_sec=1.0))

    inv = sf.inventory_drift_stats(g, cfg(inventory_horizon_sec=300))

    assert inv["horizon_sec"] == 300
    assert inv["std_bps"] > 0 and inv["p90_bps"] > inv["p10_bps"]


# ============================================================
# Bucketing must not splice non-adjacent days
# ============================================================

def test_hour_buckets_do_not_splice_across_days():
    """Grouping the grid by hour BEFORE computing markouts would make the end of
    one day's hour-00 adjacent to the next day's, reading a ~23 h mid move as a
    1 s markout. Buckets must subset a full-grid markout frame instead."""
    c = cfg(grid_staleness_sec=1.0, horizons_s=(1,))
    # Two days, each with a flat book for the first 10 s of hour 00, but at very
    # different price levels: a spliced window would see a ~100x mid jump.
    parts, days = [], ("2026-10-05", "2026-10-06")
    for day, px in zip(days, (100.0, 200.0)):
        d = pd.Timestamp(day, tz="UTC")
        q = pd.DataFrame(
            {"bid": [px - 0.01] * 10, "bid_size": [10.0] * 10,
             "ask": [px + 0.01] * 10, "ask_size": [10.0] * 10},
            index=pd.DatetimeIndex([d + pd.Timedelta(seconds=i) for i in range(10)]))
        parts.append(sf.build_second_grid(q, d, c))
    grid = pd.concat(parts).sort_index()

    v = sf.analyze_venue(grid, "okx", "NEAR", c)
    hour0 = v["by_hour"][0]
    whole = v["horizons"][1]

    # The 100 -> 200 jump happens across a 23h59m gap, which is NaN on the grid,
    # so it must never appear as a markout in any bucket.
    # 10 quotes/day, each covering its own second plus one more via the 1 s
    # tolerance -> 11 valid grid points per day.
    assert hour0["n_grid"] == 22
    assert np.isnan(hour0["mean_adverse_cost_bps"]) or \
        abs(hour0["mean_adverse_cost_bps"]) < 100
    assert np.isnan(whole["mean_adverse_cost_bps"]) or \
        abs(whole["mean_adverse_cost_bps"]) < 100
    assert hour0["sigma_mid_bps"] < 100     # not a 5,000,000 bps "return"


def test_bucket_stats_subset_matches_a_manual_mask():
    """A bucket over the whole grid must equal the ungrouped summary."""
    rows = [(i, 99.90, 10.0, 100.10, 10.0) if i < 30 else (i, 99.70, 10.0, 99.90, 10.0)
            for i in range(60)]
    c = cfg(grid_staleness_sec=1.0, horizons_s=(1,))
    g = sf.build_second_grid(quotes(rows), DAY, c)
    mk = sf.markout_frame(g, 1, 1)
    all_rows = np.ones(len(g), dtype=bool)

    bucket = sf._bucket_stats(g, mk, all_rows, "okx", c, 1)
    whole = sf.horizon_stats(g, 1, "okx", c, bucket["net_half_spread_bps"])

    assert bucket["mean_adverse_cost_bps"] == pytest.approx(whole["mean_adverse_cost_bps"])
    assert bucket["adverse_per_hour"] == pytest.approx(whole["adverse_per_hour"])
    assert bucket["edge_ratio"] == pytest.approx(whole["edge_ratio"])


def test_summarize_markout_row_mask_selects_a_subset():
    rows = [(i, 99.90, 10.0, 100.10, 10.0) if i < 30 else (i, 99.70, 10.0, 99.90, 10.0)
            for i in range(60)]
    c = cfg(grid_staleness_sec=1.0)
    g = sf.build_second_grid(quotes(rows), DAY, c)
    mk = sf.markout_frame(g, 1, 1)

    early = sf.summarize_markout(mk, 1, "okx", c, 8.0, g.index < DAY + pd.Timedelta(seconds=20))
    late = sf.summarize_markout(mk, 1, "okx", c, 8.0, g.index >= DAY + pd.Timedelta(seconds=20))

    assert early["n_adverse_events"] == 0        # the step is at t=30
    assert late["n_adverse_events"] == 1


# ============================================================
# Masking
# ============================================================

def test_mask_removes_the_window_from_the_grid(tmp_path):
    write_day(tmp_path, "20261005", csv_rows("20261005", "NEAR", 60))
    masked = ("2026-10-05 00:00:10", "2026-10-05 00:00:30")

    grids = sf.stream_grid(tmp_path, "NEAR",
                           cfg(mask_windows=(masked,), grid_staleness_sec=1.0))

    g = grids["okx"]
    window = (g.index >= pd.Timestamp(masked[0], tz="UTC")) & \
             (g.index < pd.Timestamp(masked[1], tz="UTC"))
    assert g.loc[window, "mid"].notna().sum() == 0      # masked out
    assert np.isfinite(g["mid"].iloc[5])                # outside, intact
    assert np.isfinite(g["mid"].iloc[30])               # mask is half-open at the end


def test_mask_is_exact_at_the_leading_edge_despite_the_asof_tolerance():
    """Without an explicit grid-level mask, the quote BEFORE a window would cover
    its first `grid_staleness_sec` seconds via the as-of tolerance."""
    q = flat_quotes(60)
    masked = ("2026-10-05 00:00:10", "2026-10-05 00:00:30")
    c = cfg(mask_windows=(masked,), grid_staleness_sec=5.0)

    g = sf.build_second_grid(q, DAY, c)

    # t=10..14 are within 5 s of the t=9 quote, so the tolerance alone would
    # have filled them; the grid-level mask must still null them.
    assert g["mid"].iloc[10:30].notna().sum() == 0
    assert np.isfinite(g["mid"].iloc[9])


# ============================================================
# Streaming equals single-frame
# ============================================================

def test_streaming_grid_equals_single_frame_grid(tmp_path):
    """Per-day streaming must tile into the same grid a one-shot build would give."""
    for day in ("20261005", "20261006"):
        write_day(tmp_path, day, csv_rows(day, "NEAR", 120))
    c = cfg(grid_staleness_sec=1.0)

    grids = sf.stream_grid(tmp_path, "NEAR", c)

    # Rebuild each day independently and concatenate: same index, same values.
    manual = pd.concat([
        sf.build_second_grid(
            pd.DataFrame(
                {"bid": [99.99] * 120, "bid_size": [10.0] * 120,
                 "ask": [100.01] * 120, "ask_size": [10.0] * 120},
                index=pd.DatetimeIndex(
                    [pd.Timestamp(day, tz="UTC") + pd.Timedelta(seconds=i)
                     for i in range(120)])),
            pd.Timestamp(day, tz="UTC"), c)
        for day in ("20261005", "20261006")
    ]).sort_index()

    pd.testing.assert_frame_equal(grids["okx"], manual)
    assert len(grids["okx"]) == 2 * 86_400
    # The two days are contiguous: horizon windows cross midnight.
    assert (grids["okx"].index.to_series().diff().dropna()
            == pd.Timedelta(seconds=1)).all()


def test_stream_grid_handles_a_missing_asset(tmp_path):
    write_day(tmp_path, "20261005", csv_rows("20261005", "NEAR", 30))

    grids = sf.stream_grid(tmp_path, "DOGE", cfg())

    assert len(grids["okx"]) == 86_400
    assert grids["okx"]["mid"].notna().sum() == 0


# ============================================================
# End-to-end shape + formatting
# ============================================================

def test_analyze_asset_and_formatters(tmp_path):
    for day in ("20261005", "20261006"):
        write_day(tmp_path, day, csv_rows(day, "NEAR", 600, bid=99.90, ask=100.10))
    c = cfg(grid_staleness_sec=1.0, horizons_s=(1, 5))

    res = sf.analyze_asset(tmp_path, "NEAR", c)

    assert set(res["venues"]) == {"kraken", "okx"}
    okx = res["venues"]["okx"]
    assert okx["spread"]["median_spread_bps"] == pytest.approx(20.0)
    assert 1 in okx["horizons"] and 5 in okx["horizons"]
    assert "by_hour" in okx and "by_week" in okx

    summary = sf.format_summary({"NEAR": res}, c)
    assert "NEAR" in summary and "kraken!" in summary   # floor flag at tau=1s
    assert "not daily dollars" in summary
    assert sf.format_detail("NEAR", res, c).count("net half-spread") == 2
    assert "hour-of-day" in sf.format_hour_table("NEAR", res, c)
