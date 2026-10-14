"""
tests/test_executable.py - Unit tests for
src/strategies/dislocation/executable.py

Synthetic quotes, no network. Guards: the as-of join excludes stale quotes (never
forward-fills), the executable gap math + sign are exact, taker fees net correctly,
excursions are counted on the net>0 series, dollar capacity is size-capped, and a
feed-outage window is flagged (not read as a quiet market).

Run:
    pytest tests/test_executable.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.dislocation import executable as ex  # noqa: E402


# ============================================================
# Builders
# ============================================================

def quotes(ts_ms, bid, ask, bid_size=1000.0, ask_size=1000.0):
    """Build a venue quote frame indexed by ts_recv (ms epochs)."""
    n = len(ts_ms)
    def col(v):
        return np.full(n, v, dtype=float) if np.ndim(v) == 0 else np.asarray(v, dtype=float)
    idx = pd.to_datetime(np.asarray(ts_ms), unit="ms", utc=True)
    return pd.DataFrame({"bid": col(bid), "ask": col(ask),
                         "bid_size": col(bid_size), "ask_size": col(ask_size)}, index=idx)


CFG = ex.ExecutableConfig(staleness_tolerance="500ms", kraken_fee_bps=5.0, okx_fee_bps=5.0)


# ============================================================
# As-of alignment / staleness
# ============================================================

class TestAsOfJoin:

    def test_aligns_within_tolerance(self):
        base = 1_700_000_000_000
        # Interleaved by 100ms (< 500ms tol). Backward join: the earliest union ts
        # has no prior OKX quote (dropped); the other 3 have both venues fresh.
        k = quotes([base, base + 200], 100.0, 100.1)
        o = quotes([base + 100, base + 300], 100.0, 100.1)
        m = ex.build_executable_series(k, o, CFG)
        assert len(m) == 3
        assert {"net_best_bps", "cap_best_usd", "mid"}.issubset(m.columns)

    def test_stale_quote_excluded_not_ffilled(self):
        base = 1_700_000_000_000
        k = quotes([base, base + 5000], 100.0, 100.1)  # kraken updates at 0s and 5s
        o = quotes([base], 100.0, 100.1)               # okx only at 0s
        m = ex.build_executable_series(k, o, CFG)
        # At t=5s, OKX quote is 5s old (> 500ms) -> excluded, NOT carried forward.
        # Only the t=0 timestamp (both venues fresh) survives.
        assert len(m) == 1
        assert m.index.max() < pd.to_datetime(base + 500, unit="ms", utc=True)


# ============================================================
# Gap math + fees + sign
# ============================================================

class TestGapMath:

    def test_executable_gap_and_fee_netting(self):
        base = 1_700_000_000_000
        # Kraken bid 100.5, OKX ask 100.0 -> sell_kraken_buy_okx gross = 50 bps.
        k = quotes([base], 100.5, 100.6)
        o = quotes([base], 99.9, 100.0)
        m = ex.build_executable_series(k, o, CFG)
        mid = (100.5 + 100.6 + 99.9 + 100.0) / 4.0
        gross = 10000.0 * (100.5 - 100.0) / mid
        assert m["gross_sell_k_buy_o_bps"].iloc[0] == pytest.approx(gross)
        # Net subtracts both taker legs (5 + 5 = 10 bps).
        assert m["net_sell_k_buy_o_bps"].iloc[0] == pytest.approx(gross - 10.0)
        assert m["net_best_bps"].iloc[0] == pytest.approx(gross - 10.0)

    def test_other_direction_sign(self):
        base = 1_700_000_000_000
        # OKX bid 100.5 vs Kraken ask 100.0 -> sell_okx_buy_kraken positive.
        k = quotes([base], 99.9, 100.0)
        o = quotes([base], 100.5, 100.6)
        m = ex.build_executable_series(k, o, CFG)
        assert m["net_sell_o_buy_k_bps"].iloc[0] > m["net_sell_k_buy_o_bps"].iloc[0]


# ============================================================
# Excursions on the net series + capacity
# ============================================================

class TestExecutableExcursions:

    def test_counts_net_positive_excursions_and_capacity(self):
        base = 1_700_000_000_000
        ts = [base + i * 1000 for i in range(6)]
        # Kraken bid spikes above OKX ask+fees at rows 1 and 4 (two excursions).
        kbid = [100.0, 101.5, 100.0, 100.0, 101.5, 100.0]
        k = quotes(ts, bid=kbid, ask=[b + 0.1 for b in kbid], bid_size=500.0, ask_size=500.0)
        o = quotes(ts, bid=99.9, ask=100.0, bid_size=800.0, ask_size=300.0)
        a = ex.analyze_executable(k, o, ex.ExecutableConfig(outage_gap_sec=10.0))
        assert a["executable"]["n_excursions"] == 2
        assert a["executable"]["mean_peak_net_bps"] > 0
        # Winning direction is sell-kraken/buy-okx: cap = min(k_bid_size 500, o_ask_size 300)*mid.
        expected_cap = 300.0 * ((101.5 + 101.6 + 99.9 + 100.0) / 4.0)
        assert a["executable"]["mean_capacity_usd"] == pytest.approx(expected_cap, rel=1e-3)

    def test_no_edge_when_fees_exceed_gap(self):
        base = 1_700_000_000_000
        # 3 bps raw gap < 10 bps fees -> no executable excursion (>=3 aligned rows).
        ts = [base, base + 1000, base + 2000]
        k = quotes(ts, 100.03, 100.13)
        o = quotes(ts, 99.93, 100.0)
        a = ex.analyze_executable(k, o, CFG)
        assert a["executable"]["n_excursions"] == 0


# ============================================================
# Feed-outage flagging
# ============================================================

class TestOutages:

    def test_outage_flagged_not_treated_as_quiet(self):
        base = 1_700_000_000_000
        # A 60s gap in the middle -> one outage, excluded from the active-time base.
        ts = [base, base + 1000, base + 61_000, base + 62_000]
        k = quotes(ts, 100.0, 100.1)
        o = quotes(ts, 100.0, 100.1)
        a = ex.analyze_executable(k, o, ex.ExecutableConfig(outage_gap_sec=10.0))
        assert a["outages"]["n_outages"] == 1
        assert a["outages"]["outage_seconds"] == pytest.approx(59.0, abs=1.0)
        # Active seconds excludes the outage, so it is not counted as quiet market.
        assert a["coverage"]["active_seconds"] < a["coverage"]["covered_seconds"]


# ============================================================
# Compare to Phase A
# ============================================================

class TestCompare:

    def test_ratio(self):
        r = ex.compare_to_candle(executable_per_day=2.0, candle_per_day=40.0)
        assert r["ratio"] == pytest.approx(0.05)

    def test_ratio_nan_when_no_candle(self):
        r = ex.compare_to_candle(1.0, 0.0)
        assert np.isnan(r["ratio"])


# ============================================================
# Size units: OKX quotes contracts, Kraken quotes base currency
# ============================================================
#
# `cap_best_usd` takes a min() ACROSS the two venues' sizes, so both legs must be
# converted to base units FIRST. Comparing them raw mixed units and was a real
# bug in the Phase B capacity figures (OKX BTC notional came out 100x large).

class TestSizeUnits:

    def test_okx_contracts_are_converted_before_the_min(self):
        """The regression test: OKX size x ctVal must happen BEFORE min()."""
        base = 1_700_000_000_000
        ts = [base, base + 1000, base + 2000]
        # Raw sizes: kraken 5 (base units), okx 100 (contracts).
        # With ctVal = 0.01 the OKX leg is 1.0 base units -> OKX binds, not Kraken.
        # Taking min() on the RAW numbers would pick kraken's 5 and overstate 5x.
        k = quotes(ts, bid=101.5, ask=101.6, bid_size=5.0, ask_size=5.0)
        o = quotes(ts, bid=99.9, ask=100.0, bid_size=100.0, ask_size=100.0)
        cfg = ex.ExecutableConfig(staleness_tolerance="500ms", okx_size_mult=0.01)

        m = ex.build_executable_series(k, o, cfg)

        mid = (101.5 + 101.6 + 99.9 + 100.0) / 4.0
        assert m["cap_best_usd"].iloc[-1] == pytest.approx(1.0 * mid, rel=1e-9)
        assert bool(m["okx_leg_binds"].iloc[-1]) is True

    def test_kraken_leg_can_bind_after_conversion(self):
        base = 1_700_000_000_000
        ts = [base, base + 1000, base + 2000]
        # kraken 2 base units vs okx 1000 contracts x 0.01 = 10 base units.
        k = quotes(ts, bid=101.5, ask=101.6, bid_size=2.0, ask_size=2.0)
        o = quotes(ts, bid=99.9, ask=100.0, bid_size=1000.0, ask_size=1000.0)
        cfg = ex.ExecutableConfig(staleness_tolerance="500ms", okx_size_mult=0.01)

        m = ex.build_executable_series(k, o, cfg)

        mid = (101.5 + 101.6 + 99.9 + 100.0) / 4.0
        assert m["cap_best_usd"].iloc[-1] == pytest.approx(2.0 * mid, rel=1e-9)
        assert bool(m["okx_leg_binds"].iloc[-1]) is False

    def test_multiplier_scales_capacity_linearly(self):
        base = 1_700_000_000_000
        ts = [base, base + 1000, base + 2000]
        k = quotes(ts, bid=101.5, ask=101.6, bid_size=1e9, ask_size=1e9)  # never binds
        o = quotes(ts, bid=99.9, ask=100.0, bid_size=50.0, ask_size=50.0)

        one = ex.build_executable_series(
            k, o, ex.ExecutableConfig(staleness_tolerance="500ms", okx_size_mult=1.0))
        ten = ex.build_executable_series(
            k, o, ex.ExecutableConfig(staleness_tolerance="500ms", okx_size_mult=10.0))

        assert ten["cap_best_usd"].iloc[-1] == pytest.approx(
            10.0 * one["cap_best_usd"].iloc[-1], rel=1e-9)

    def test_units_do_not_affect_the_gap_or_excursion_count(self):
        """Capacity is a size metric; the bps edge must be untouched by ctVal."""
        base = 1_700_000_000_000
        ts = [base + i * 1000 for i in range(6)]
        kbid = [100.0, 101.5, 100.0, 100.0, 101.5, 100.0]
        k = quotes(ts, bid=kbid, ask=[b + 0.1 for b in kbid], bid_size=500.0, ask_size=500.0)
        o = quotes(ts, bid=99.9, ask=100.0, bid_size=800.0, ask_size=300.0)

        a = ex.analyze_executable(k, o, ex.ExecutableConfig(outage_gap_sec=10.0))
        b = ex.analyze_executable(
            k, o, ex.ExecutableConfig(outage_gap_sec=10.0, okx_size_mult=10.0))

        assert a["executable"]["n_excursions"] == b["executable"]["n_excursions"]
        assert a["net_gap_bps"] == b["net_gap_bps"]
        assert b["executable"]["mean_capacity_usd"] > a["executable"]["mean_capacity_usd"]

    @pytest.mark.parametrize("asset,expected", [
        ("BTC", 0.01), ("NEAR", 10.0), ("UNI", 1.0), ("ARB", 10.0),
    ])
    def test_for_asset_resolves_the_published_ctval(self, asset, expected):
        """ctVal from GET /api/v5/public/instruments?instType=SWAP, read 2026-10-14."""
        cfg = ex.ExecutableConfig.for_asset(asset)

        assert cfg.okx_size_mult == expected
        assert cfg.kraken_size_mult == 1.0

    def test_for_asset_keeps_other_overrides(self):
        cfg = ex.ExecutableConfig.for_asset("NEAR", kraken_fee_bps=3.0, okx_fee_bps=3.0)

        assert cfg.okx_size_mult == 10.0
        assert cfg.kraken_fee_bps == 3.0

    def test_default_config_is_unconverted(self):
        """Bare ExecutableConfig stays 1.0/1.0 so synthetic base-unit tests hold."""
        cfg = ex.ExecutableConfig()

        assert cfg.kraken_size_mult == 1.0 and cfg.okx_size_mult == 1.0
