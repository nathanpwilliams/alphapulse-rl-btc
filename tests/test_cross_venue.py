"""
tests/test_cross_venue.py - Unit tests for
src/strategies/funding/cross_venue.py

Synthetic data, no network. The properties under test are the ones that make or
break the cross-venue measurement: the two venues must align onto a common hourly
grid without look-ahead, the per-hour spread must be the hand-computed difference,
sign-flip frequency must be counted correctly, the shared cost helpers must match
the single-venue path, and correlation must behave.

Run:
    pytest tests/test_cross_venue.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.funding import cross_venue as cv  # noqa: E402
from src.strategies.funding.analyzer import (  # noqa: E402
    _break_even,
    _net_by_holding_period,
)


# ============================================================
# Synthetic builders
# ============================================================

def kraken_hourly(rate, n=500, start="2024-01-01") -> pd.DataFrame:
    """Kraken-style per-hour funding: hourly cadence, `funding_rate` column."""
    rate = np.asarray(rate, dtype=float) if np.ndim(rate) else np.full(n, rate)
    idx = pd.date_range(start, periods=len(rate), freq="1h", tz="UTC")
    return pd.DataFrame({"funding_rate": rate}, index=idx)


def okx_periodic(hourly_rate, n=100, interval_h=8, start="2024-01-01") -> pd.DataFrame:
    """OKX-style frame already carrying `funding_rate_hourly` (per-hour).

    Settlements are `interval_h` apart; `funding_rate_hourly` is the per-hour
    value the loader would have produced (period rate / interval).
    """
    hourly_rate = (
        np.asarray(hourly_rate, dtype=float) if np.ndim(hourly_rate) else np.full(n, hourly_rate)
    )
    idx = pd.date_range(start, periods=len(hourly_rate), freq=f"{interval_h}h", tz="UTC")
    return pd.DataFrame(
        {
            "funding_rate": hourly_rate * interval_h,  # period rate
            "interval_hours": float(interval_h),
            "funding_rate_hourly": hourly_rate,
        },
        index=idx,
    )


# ============================================================
# Alignment
# ============================================================

class TestAlignment:

    def test_aligns_different_cadences_onto_hourly_grid(self):
        k = kraken_hourly(1e-5, n=200)          # hourly
        o = okx_periodic(2e-6, n=30, interval_h=8)  # every 8h
        aligned = cv.align_funding_series(k, o)

        # Result is on an hourly grid.
        assert isinstance(aligned.index, pd.DatetimeIndex)
        diffs = aligned.index.to_series().diff().dropna().dt.total_seconds() / 3600.0
        assert np.allclose(diffs, 1.0)
        assert set(aligned.columns) == {"kraken_hourly", "okx_hourly", "spread_hourly"}
        assert not aligned.isna().any().any()

    def test_okx_forward_filled_across_its_interval(self):
        # OKX rate changes every 8h; it must hold constant for the 8 hours between.
        k = kraken_hourly(0.0, n=48)
        rates = np.array([1e-6, 5e-6, 9e-6])
        o = okx_periodic(rates, interval_h=8)
        aligned = cv.align_funding_series(k, o)

        # The first 8 aligned hours should all carry the first OKX rate.
        first_block = aligned["okx_hourly"].iloc[:8].to_numpy()
        assert np.allclose(first_block, 1e-6)

    def test_no_lookahead_uses_past_settlement_only(self):
        """At any hour, the OKX value must be the most recent settlement AT or
        BEFORE that hour, never a future one."""
        k = kraken_hourly(0.0, n=24)
        # OKX settlements at 00:00 (=2e-6) and 08:00 (=7e-6).
        o = okx_periodic(np.array([2e-6, 7e-6]), interval_h=8)
        aligned = cv.align_funding_series(k, o)

        # Hour 04:00 is before the 08:00 settlement => must still be 2e-6.
        h04 = pd.Timestamp("2024-01-01 04:00", tz="UTC")
        assert aligned.loc[h04, "okx_hourly"] == pytest.approx(2e-6)
        # Hour 08:00 gets the new rate.
        h08 = pd.Timestamp("2024-01-01 08:00", tz="UTC")
        assert aligned.loc[h08, "okx_hourly"] == pytest.approx(7e-6)

    def test_only_overlap_analyzed(self):
        # Kraken spans Jan 1-10; OKX starts Jan 3. Overlap begins Jan 3.
        k = kraken_hourly(1e-5, n=24 * 10, start="2024-01-01")
        o = okx_periodic(2e-6, n=40, interval_h=8, start="2024-01-03")
        aligned = cv.align_funding_series(k, o)
        assert aligned.index[0] >= pd.Timestamp("2024-01-03", tz="UTC")

    def test_no_overlap_raises(self):
        k = kraken_hourly(1e-5, n=48, start="2024-01-01")
        o = okx_periodic(2e-6, n=10, interval_h=8, start="2024-06-01")
        with pytest.raises(ValueError, match="do not overlap"):
            cv.align_funding_series(k, o)

    def test_missing_column_raises(self):
        k = kraken_hourly(1e-5, n=48)
        o = okx_periodic(2e-6, n=10).drop(columns=["funding_rate_hourly"])
        with pytest.raises(ValueError, match="funding_rate_hourly"):
            cv.align_funding_series(k, o)


# ============================================================
# Spread math
# ============================================================

class TestSpreadMath:

    CFG = cv.CrossVenueConfig()

    def test_constant_spread_equals_difference(self):
        k_rate, o_rate = 1e-5, 3e-6
        k = kraken_hourly(k_rate, n=400)
        o = okx_periodic(o_rate, n=60, interval_h=8)
        stats = cv.analyze_spread(k, o, self.CFG)
        # spread = kraken_hourly - okx_hourly, constant here.
        assert stats["spread"]["mean_hourly"] == pytest.approx(k_rate - o_rate)
        assert stats["spread"]["min_hourly"] == pytest.approx(k_rate - o_rate)
        assert stats["spread"]["max_hourly"] == pytest.approx(k_rate - o_rate)

    def test_annualized_spread(self):
        k_rate, o_rate = 1e-5, 3e-6
        k = kraken_hourly(k_rate, n=400)
        o = okx_periodic(o_rate, n=60, interval_h=8)
        stats = cv.analyze_spread(k, o, self.CFG)
        expected = (k_rate - o_rate) * 8760 * 100
        assert stats["spread"]["annualized_mean_pct"] == pytest.approx(expected)

    def test_mean_abs_spread_bps(self):
        k = kraken_hourly(1e-5, n=400)
        o = okx_periodic(3e-6, n=60, interval_h=8)
        stats = cv.analyze_spread(k, o, self.CFG)
        expected_bps = abs(1e-5 - 3e-6) * 10_000
        assert stats["magnitude"]["mean_abs_spread_bps_per_hour"] == pytest.approx(expected_bps)


# ============================================================
# Sign-flip stability (the key metric)
# ============================================================

class TestSignStability:

    def test_stable_positive_spread_no_flips(self):
        k = kraken_hourly(1e-5, n=400)
        o = okx_periodic(3e-6, n=60, interval_h=8)  # kraken > okx always => spread > 0
        stats = cv.analyze_spread(k, o, cv.CrossVenueConfig())
        ss = stats["sign_stability"]
        assert ss["frac_positive"] == pytest.approx(1.0)
        assert ss["n_flips"] == 0
        assert ss["mean_hours_between_flips"] == float("inf")

    def test_spread_flips_every_hour(self):
        # Kraken alternates sign each hour; OKX flat 0 => spread flips each hour.
        n = 240
        alt = np.where(np.arange(n) % 2 == 0, 1e-5, -1e-5)
        k = kraken_hourly(alt, n=n)
        o = okx_periodic(0.0, n=n // 8 + 2, interval_h=8)
        stats = cv.analyze_spread(k, o, cv.CrossVenueConfig())
        assert stats["sign_stability"]["flip_fraction"] == pytest.approx(1.0)
        assert stats["sign_stability"]["mean_hours_between_flips"] == pytest.approx(1.0)

    def test_spread_flips_every_n_hours(self):
        # Kraken holds a sign for 6h then flips; OKX flat 0.
        n = 240
        block = 6
        signs = np.where((np.arange(n) // block) % 2 == 0, 1e-5, -1e-5)
        k = kraken_hourly(signs, n=n)
        o = okx_periodic(0.0, n=n // 8 + 2, interval_h=8)
        stats = cv.analyze_spread(k, o, cv.CrossVenueConfig())
        # A flip every 6 hours => mean hours between flips ~ 6.
        assert stats["sign_stability"]["mean_hours_between_flips"] == pytest.approx(6.0, rel=0.1)


# ============================================================
# Shared-helper reuse (regression guard on the refactor)
# ============================================================

class TestSharedHelperReuse:

    def test_break_even_matches_helper(self):
        k = kraken_hourly(1e-5, n=400)
        o = okx_periodic(3e-6, n=60, interval_h=8)
        cfg = cv.CrossVenueConfig(round_trip_cost_bps=15.0)
        stats = cv.analyze_spread(k, o, cfg)

        mean_abs = stats["magnitude"]["mean_abs_spread_hourly"]
        expected = _break_even(mean_abs, 15.0)
        assert stats["cost_hurdle"]["break_even_hours"] == pytest.approx(expected["break_even_hours"])

    def test_net_table_matches_helper(self):
        k = kraken_hourly(1e-5, n=400)
        o = okx_periodic(3e-6, n=60, interval_h=8)
        cfg = cv.CrossVenueConfig(round_trip_cost_bps=20.0)
        stats = cv.analyze_spread(k, o, cfg)

        mean_abs = stats["magnitude"]["mean_abs_spread_hourly"]
        expected = _net_by_holding_period(mean_abs, 20.0, cfg.holding_periods_hours)
        for h in cfg.holding_periods_hours:
            assert stats["net_by_holding_period"][h]["net_bps"] == pytest.approx(expected[h]["net_bps"])

    def test_break_even_scales_with_cost(self):
        k = kraken_hourly(1e-5, n=400)
        o = okx_periodic(3e-6, n=60, interval_h=8)
        s10 = cv.analyze_spread(k, o, cv.CrossVenueConfig(round_trip_cost_bps=10.0))
        s20 = cv.analyze_spread(k, o, cv.CrossVenueConfig(round_trip_cost_bps=20.0))
        assert s20["cost_hurdle"]["break_even_hours"] == pytest.approx(
            2.0 * s10["cost_hurdle"]["break_even_hours"]
        )


# ============================================================
# Correlation
# ============================================================

class TestCorrelation:

    def test_perfectly_correlated_series(self):
        # Both venues driven by the same varying signal => correlation ~ 1.0.
        n = 400
        signal = 1e-5 * np.sin(np.linspace(0, 20, n))
        k = kraken_hourly(signal, n=n)
        # OKX carries the same signal sampled every 8h (per-hour basis).
        o = okx_periodic(signal[::8], interval_h=8)
        stats = cv.analyze_spread(k, o, cv.CrossVenueConfig())
        assert stats["correlation"] == pytest.approx(1.0, abs=0.05)

    def test_summary_renders(self):
        k = kraken_hourly(1e-5, n=400)
        o = okx_periodic(3e-6, n=60, interval_h=8)
        stats = cv.analyze_spread(k, o, cv.CrossVenueConfig())
        text = cv.format_summary(stats)
        assert "CROSS-VENUE FUNDING SPREAD" in text
        assert "sign stability" in text.lower()
