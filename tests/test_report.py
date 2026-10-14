"""
tests/test_report.py - Unit tests for src/strategies/dislocation/report.py

Synthetic excursion records and per-day stats; no disk, no network (only
`collect()` touches files, and that is exercised by tests/test_streaming.py).
Guards the arithmetic that the Phase B headline figures depend on: partial-day
rate normalisation, the <=1 s resolution-floor split, size-capped capacity
percentiles, the direction split, weekday/weekend bucketing, and the Phase A
ratio -- plus that `format_report` renders every asset without raising.

Run:
    pytest tests/test_report.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.dislocation import report as rp  # noqa: E402


# ============================================================
# Builders
# ============================================================

def rec(day: str, hour: int = 0, duration: float = 2.0, net_bps: float = 10.0,
        cap_usd: float = 1000.0, direction: str = "sell_k_buy_o") -> dict:
    """One excursion record in the shape `streaming.asset_day_records` emits."""
    return {
        "day": day,
        "start": pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=hour),
        "hour": hour,
        "weekday": pd.Timestamp(day).weekday(),
        "duration_sec": duration,
        "n_updates": 3,
        "net_bps": net_bps,
        "cap_usd": cap_usd,
        "mid": 100.0,
        "capturable_usd": cap_usd * net_bps / 10000.0,
        "direction": direction,
    }


def day_stats(records: list[dict], active_hours: float, best_net_bps: float = 5.0,
              n_aligned: int = 1000) -> dict:
    """Per-day stats in the shape `streaming.analyze_streaming` emits."""
    return {
        "n_aligned": n_aligned,
        "covered_seconds": active_hours * 3600.0,
        "active_seconds": active_hours * 3600.0,
        "outage_seconds": 0.0,
        "masked_seconds": 0.0,
        "n_outages": 0,
        "best_net_bps": best_net_bps,
        "n_excursions": len(records),
        "capturable_usd": float(sum(r["capturable_usd"] for r in records)),
    }


# ============================================================
# Rate normalisation (the partial-day guard)
# ============================================================

def test_rate_per_day_scales_a_partial_day_up():
    # $50 earned over 12 clean hours is a $100/day rate.
    assert rp.rate_per_day(50.0, 12 * 3600.0) == pytest.approx(100.0)
    # A full day is unchanged.
    assert rp.rate_per_day(50.0, 24 * 3600.0) == pytest.approx(50.0)


def test_rate_per_day_zero_active_is_nan():
    assert np.isnan(rp.rate_per_day(10.0, 0.0))


def test_daily_table_rate_normalises_partial_days():
    full = [rec("20261005", cap_usd=1000.0, net_bps=10.0)]      # $1.00
    part = [rec("20261013", cap_usd=1000.0, net_bps=10.0)]      # $1.00 in 6 h
    per_day = {"20261005": day_stats(full, 24.0), "20261013": day_stats(part, 6.0)}

    daily = rp.daily_table(per_day)

    assert daily["20261005"]["usd_per_day"] == pytest.approx(1.0)
    assert daily["20261013"]["usd_per_day"] == pytest.approx(4.0)   # 1.0 / 6 * 24
    assert daily["20261013"]["capturable_usd"] == pytest.approx(1.0)  # raw untouched
    assert daily["20261013"]["exc_per_day"] == pytest.approx(4.0)


def test_daily_summary_uses_the_rate_series():
    per_day = {"20261005": day_stats([rec("20261005")], 24.0),
               "20261013": day_stats([rec("20261013")], 6.0)}

    s = rp.daily_summary(rp.daily_table(per_day), "usd_per_day")

    assert s["min"] == pytest.approx(1.0)
    assert s["max"] == pytest.approx(4.0)
    assert s["mean"] == pytest.approx(2.5)


# ============================================================
# Coverage
# ============================================================

def test_coverage_sums_hours_and_outages():
    a = day_stats([], 24.0)
    b = day_stats([], 22.6)
    b["outage_seconds"] = 360.0
    b["masked_seconds"] = 397.0
    b["n_outages"] = 1

    c = rp.coverage({"20261005": a, "20261002": b})

    assert c["clean_hours"] == pytest.approx(46.6)
    assert c["outage_hours"] == pytest.approx(0.1)
    assert c["masked_seconds"] == pytest.approx(397.0)
    assert c["n_outages"] == 1


# ============================================================
# Durations and the resolution floor
# ============================================================

def test_duration_stats_splits_at_the_resolution_floor():
    recs = ([rec("20261005", duration=0.2)] * 3
            + [rec("20261005", duration=1.0)]          # exactly at the floor -> "<="
            + [rec("20261005", duration=2.0)]
            + [rec("20261005", duration=90.0)])

    d = rp.duration_stats(recs, floor_sec=1.0)

    assert d["n"] == 6
    assert d["n_le_floor"] == 4 and d["n_gt_floor"] == 2
    assert d["frac_le_floor"] == pytest.approx(4 / 6)
    assert d["n_gt_5s"] == 1 and d["n_gt_60s"] == 1
    assert d["max"] == pytest.approx(90.0)


def test_duration_stats_empty():
    assert rp.duration_stats([])["n"] == 0


# ============================================================
# Capacity / net bps
# ============================================================

def test_capacity_stats_percentiles():
    recs = [rec("20261005", cap_usd=c) for c in (10, 20, 30, 40, 1000)]

    c = rp.capacity_stats(recs)

    assert c["median"] == pytest.approx(30.0)
    assert c["max"] == pytest.approx(1000.0)
    assert c["mean"] == pytest.approx(220.0)


def test_capturable_is_cap_times_net_over_1e4():
    r = rec("20261005", cap_usd=50_000.0, net_bps=4.0)
    assert r["capturable_usd"] == pytest.approx(20.0)


def test_net_bps_stats():
    b = rp.net_bps_stats([rec("20261005", net_bps=x) for x in (1.0, 2.0, 3.0)])
    assert b["median"] == pytest.approx(2.0) and b["max"] == pytest.approx(3.0)


# ============================================================
# Hour / weekday-weekend / direction
# ============================================================

def test_hour_histogram_has_24_rows_and_bins_by_hour():
    recs = [rec("20261005", hour=3), rec("20261005", hour=3), rec("20261005", hour=20)]

    h = rp.hour_histogram(recs)

    assert len(h) == 24
    assert h[3]["n"] == 2 and h[20]["n"] == 1 and h[0]["n"] == 0
    assert h[3]["usd"] == pytest.approx(2 * recs[0]["capturable_usd"])


@pytest.mark.parametrize("day,weekend", [
    ("20261003", True),    # Sat
    ("20261004", True),    # Sun
    ("20261005", False),   # Mon
    ("20261002", False),   # Fri
])
def test_is_weekend(day, weekend):
    assert rp.is_weekend(day) is weekend


def test_weekday_weekend_split_uses_rates():
    per_day = {
        "20261005": day_stats([rec("20261005", cap_usd=1000.0, net_bps=10.0)], 24.0),  # Mon
        "20261003": day_stats([rec("20261003", cap_usd=1000.0, net_bps=10.0)] * 2, 12.0),  # Sat
    }

    wk = rp.weekday_weekend(per_day)

    assert wk["weekday"]["n_excursions"] == 1
    assert wk["weekday"]["usd_per_day"] == pytest.approx(1.0)
    assert wk["weekend"]["n_excursions"] == 2
    assert wk["weekend"]["usd_per_day"] == pytest.approx(4.0)   # $2 / 12 h * 24
    assert wk["ratio_we_wd"] == pytest.approx(4.0)


def test_direction_split_flags_one_sided():
    recs = ([rec("20261005", direction="sell_k_buy_o", net_bps=1.0)] * 9
            + [rec("20261005", direction="sell_o_buy_k", net_bps=5.0)])

    d = rp.direction_split(recs)

    assert d["sell_k_buy_o"]["n"] == 9
    assert d["sell_k_buy_o"]["frac"] == pytest.approx(0.9)
    assert d["sell_o_buy_k"]["n"] == 1
    # The minority direction carrying the larger edge is the Phase B signature.
    assert d["sell_o_buy_k"]["median_net_bps"] > d["sell_k_buy_o"]["median_net_bps"]


def test_direction_split_empty():
    assert rp.direction_split([])["n"] == 0


# ============================================================
# Phase A comparison
# ============================================================

def test_phase_a_comparison_all_and_above_floor():
    # 10 excursions over 24 clean h; 4 of them longer than 1 s.
    recs = ([rec("20261005", duration=0.5)] * 6
            + [rec("20261005", duration=3.0, cap_usd=1000.0, net_bps=10.0)] * 4)

    p = rp.phase_a_comparison(recs, clean_h=24.0, candle_per_day=5.0, floor_sec=1.0)

    assert p["executable_per_day"] == pytest.approx(10.0)
    assert p["ratio"] == pytest.approx(2.0)
    assert p["n_above_floor"] == 4
    assert p["executable_per_day_above_floor"] == pytest.approx(4.0)
    assert p["usd_per_day_above_floor"] == pytest.approx(4.0)   # 4 x $1.00
    assert p["ratio_above_floor"] == pytest.approx(0.8)


# ============================================================
# End-to-end summarize + format
# ============================================================

def _two_asset_results():
    btc_day = day_stats([], 24.0, best_net_bps=-3.0)
    arb_recs = [rec("20261005", hour=h, duration=2.0) for h in (1, 2, 15)]
    return {
        "BTC": {"asset": "BTC", "per_day": {"20261005": btc_day}, "records": []},
        "ARB": {"asset": "ARB",
                "per_day": {"20261005": day_stats(arb_recs, 24.0, best_net_bps=12.0)},
                "records": arb_recs},
    }


def test_summarize_shapes_every_asset():
    config = rp.ReportConfig(assets=("BTC", "ARB"), mask_windows=())

    s = rp.summarize(_two_asset_results(), config)

    assert set(s) == {"BTC", "ARB"}
    assert s["BTC"]["n_excursions"] == 0
    assert s["BTC"]["best_net_bps"] == pytest.approx(-3.0)
    assert s["BTC"]["total_capturable_usd"] == pytest.approx(0.0)
    assert s["ARB"]["n_excursions"] == 3
    assert s["ARB"]["exc_per_day"] == pytest.approx(3.0)
    assert s["ARB"]["phase_a"]["candle_per_day"] == pytest.approx(41.9)


def test_format_report_renders_without_raising():
    config = rp.ReportConfig(assets=("BTC", "ARB"), mask_windows=())
    s = rp.summarize(_two_asset_results(), config)

    text = rp.format_report(s, config)

    assert "PHASE B" in text
    assert "DAILY CAPTURABLE DOLLARS" in text
    assert "DIRECTION SPLIT" in text
    for asset in ("BTC", "ARB"):
        assert asset in text
    assert len(text.splitlines()) > 60      # all sections present


def test_format_report_handles_a_zero_excursion_asset():
    """The BTC-control path: no excursions must not blow up any table."""
    config = rp.ReportConfig(assets=("BTC",), mask_windows=())
    results = {"BTC": {"asset": "BTC",
                       "per_day": {"20261005": day_stats([], 24.0, best_net_bps=-3.0)},
                       "records": []}}

    text = rp.format_report(rp.summarize(results, config), config)

    assert "BTC" in text


def test_stream_config_propagates_fees_and_mask():
    config = rp.ReportConfig(assets=("BTC",), fee_bps_per_side=7.5,
                             mask_windows=(("2026-10-06 07:00:42", "2026-10-06 07:07:19"),))

    sc = config.stream_config()

    assert sc.executable.kraken_fee_bps == 7.5 and sc.executable.okx_fee_bps == 7.5
    assert len(sc.mask_windows) == 1
    w0, w1 = sc.mask_windows[0]
    assert (w1 - w0).total_seconds() == pytest.approx(397.0)
    assert str(w0.tz) == "UTC"


def test_default_mask_is_the_kraken_outage():
    """The committed Phase B numbers depend on this exact window."""
    (a, b), = rp.PHASE_B_MASK
    assert (pd.Timestamp(b, tz="UTC") - pd.Timestamp(a, tz="UTC")).total_seconds() == 397.0


# ============================================================
# Capacity units + the three committed views
# ============================================================

def test_capacity_stats_reports_the_binding_leg_fraction():
    recs = ([rec("20261005", cap_usd=10.0) | {"okx_leg_binds": True}] * 3
            + [rec("20261005", cap_usd=20.0) | {"okx_leg_binds": False}])

    c = rp.capacity_stats(recs)

    assert c["frac_okx_binds"] == pytest.approx(0.75)
    assert c["median"] == pytest.approx(10.0)


def test_capacity_stats_omits_binding_fraction_when_absent():
    """Older records without the flag must not crash the table."""
    c = rp.capacity_stats([rec("20261005", cap_usd=5.0)])

    assert "frac_okx_binds" not in c


def test_minority_direction_picks_the_less_common_side():
    recs = ([rec("20261005", direction="sell_k_buy_o")] * 9
            + [rec("20261005", direction="sell_o_buy_k")])

    assert rp.minority_direction(recs) == "sell_o_buy_k"
    assert rp.minority_direction([]) is None


def test_filtered_views_apply_duration_then_direction():
    recs = (
        # 6 sub-floor, majority direction -> excluded from both filtered views
        [rec("20261005", duration=0.5, direction="sell_k_buy_o")] * 6
        # 3 above-floor, majority -> in `above_floor` only
        + [rec("20261005", duration=3.0, direction="sell_k_buy_o", cap_usd=1000.0,
               net_bps=10.0)] * 3
        # 1 above-floor, MINORITY -> in both filtered views
        + [rec("20261005", duration=4.0, direction="sell_o_buy_k", cap_usd=2000.0,
               net_bps=10.0)]
    )

    v = rp.filtered_views(recs, clean_h=24.0, floor_sec=1.0)

    assert v["minority_direction"] == "sell_o_buy_k"
    assert v["all"]["n"] == 10
    assert v["above_floor"]["n"] == 4
    assert v["transient_above_floor"]["n"] == 1
    # $1.00 x 3 + $2.00 x 1 = $5.00 above floor; $2.00 transient.
    assert v["above_floor"]["total_usd"] == pytest.approx(5.0)
    assert v["transient_above_floor"]["total_usd"] == pytest.approx(2.0)
    assert v["transient_above_floor"]["median_cap_usd"] == pytest.approx(2000.0)


def test_filtered_views_rate_normalise_a_partial_day():
    recs = [rec("20261013", duration=3.0, cap_usd=1000.0, net_bps=10.0)]

    v = rp.filtered_views(recs, clean_h=6.0, floor_sec=1.0)

    assert v["all"]["per_day"] == pytest.approx(4.0)       # 1 / 6 h x 24
    assert v["all"]["usd_per_day"] == pytest.approx(4.0)   # $1.00 / 6 h x 24


def test_filtered_views_handle_zero_records():
    v = rp.filtered_views([], clean_h=24.0)

    assert v["all"]["n"] == 0 and v["all"]["total_usd"] == 0.0
    assert v["minority_direction"] is None


def test_summarize_includes_views_and_format_renders_them():
    config = rp.ReportConfig(assets=("ARB",), mask_windows=())
    recs = [rec("20261005", duration=2.0, direction="sell_k_buy_o"),
            rec("20261005", duration=0.5, direction="sell_o_buy_k")]
    results = {"ARB": {"asset": "ARB",
                       "per_day": {"20261005": day_stats(recs, 24.0, best_net_bps=12.0)},
                       "records": recs}}

    s = rp.summarize(results, config)
    text = rp.format_report(s, config)

    assert s["ARB"]["views"]["above_floor"]["n"] == 1
    assert "VIEWS" in text and "transient" in text
    assert "OKX leg binds" in text
