"""
tests/test_dislocation_scanner.py - Unit tests for
src/strategies/dislocation/scanner.py

Synthetic data, no network. Guards the load-bearing properties: alignment drops
(never forward-fills) unmatched bars, the gap math and sign convention are exact,
excursion detection counts runs/durations/reversion correctly, a constant premium
registers as STRUCTURAL (not transient) excursions, the AR(1) half-life is
recovered, and the scanner table has the right shape and sort order.

Run:
    pytest tests/test_dislocation_scanner.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.dislocation import scanner as ds  # noqa: E402


# ============================================================
# Synthetic builders
# ============================================================

def close_df(closes, start="2024-01-01", freq="5min") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame({"close": np.asarray(closes, dtype=float)}, index=idx)


def pair_from_gap(gap_bps, okx_price: float = 100.0, start="2024-01-01"):
    """Build (kraken_df, okx_df) whose per-bar gap_bps equals `gap_bps`."""
    gap = np.asarray(gap_bps, dtype=float)
    okx = np.full(len(gap), okx_price)
    kraken = okx * (1.0 + gap / 10000.0)
    return close_df(kraken, start), close_df(okx, start)


# ============================================================
# Alignment
# ============================================================

class TestAlignment:

    def test_jittered_timestamps_align_within_tolerance(self):
        k = close_df([100, 101, 102], start="2024-01-01 00:00:00")
        # OKX 20s later than each Kraken bar -> within 1min tolerance.
        oidx = pd.to_datetime(["2024-01-01 00:00:20", "2024-01-01 00:05:20",
                               "2024-01-01 00:10:20"]).tz_localize("UTC")
        o = pd.DataFrame({"close": [100.0, 100.0, 100.0]}, index=oidx)
        aligned = ds.align_venues(k, o, ds.DislocationConfig())
        assert len(aligned) == 3
        assert set(aligned.columns) == {"kraken_close", "okx_close", "gap_bps"}

    def test_out_of_tolerance_bars_dropped_not_ffilled(self):
        k = close_df([100, 101, 102], start="2024-01-01 00:00:00")  # :00, :05, :10
        # OKX only at :00:20 and :10:20; the :05 Kraken bar has no match within 1min.
        oidx = pd.to_datetime(["2024-01-01 00:00:20", "2024-01-01 00:10:20"]).tz_localize("UTC")
        o = pd.DataFrame({"close": [100.0, 100.0]}, index=oidx)
        aligned = ds.align_venues(k, o, ds.DislocationConfig())
        assert len(aligned) == 2  # :05 dropped, NOT forward-filled
        assert pd.Timestamp("2024-01-01 00:05:00", tz="UTC") not in aligned.index

    def test_no_overlap_raises(self):
        k = close_df([100] * 5, start="2024-01-01")
        o = close_df([100] * 5, start="2024-06-01")
        with pytest.raises(ValueError, match="do not overlap"):
            ds.align_venues(k, o, ds.DislocationConfig())


# ============================================================
# Gap math / sign convention
# ============================================================

class TestGapMath:

    def test_gap_bps_value_and_sign(self):
        # kraken 101 vs okx 100 -> +100 bps (Kraken richer).
        k, o = close_df([101.0, 99.0]), close_df([100.0, 100.0])
        aligned = ds.align_venues(k, o, ds.DislocationConfig())
        assert aligned["gap_bps"].iloc[0] == pytest.approx(100.0)
        assert aligned["gap_bps"].iloc[1] == pytest.approx(-100.0)  # Kraken cheaper


# ============================================================
# Excursion detection
# ============================================================

class TestExcursions:

    def test_find_runs(self):
        mask = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0], dtype=bool)
        assert ds._find_runs(mask) == [(2, 5), (6, 8)]

    def test_excursion_counts_durations_reversion(self):
        abs_gap = np.array([0, 0, 30, 30, 30, 0, 0, 25, 25, 0], dtype=float)
        r = ds._excursions(abs_gap, threshold=20.0, revert_level=10.0, bar_minutes=5.0)
        assert r["n_excursions"] == 2
        assert r["mean_duration_bars"] == pytest.approx(2.5)   # runs of 3 and 2
        assert r["median_duration_bars"] == pytest.approx(2.5)
        assert r["mean_duration_min"] == pytest.approx(2.5 * 5)
        assert r["mean_peak_abs_gap_bps"] == pytest.approx(27.5)  # (30 + 25) / 2
        assert r["frac_bars_in_excursion"] == pytest.approx(0.5)  # 5 of 10 bars
        # revert (below 10) from each start: run1 start@2 -> idx5 (3 bars); run2 @7 -> idx9 (2)
        assert r["mean_time_to_revert_bars"] == pytest.approx(2.5)

    def test_no_excursions(self):
        abs_gap = np.array([1, 2, 3, 2, 1], dtype=float)
        r = ds._excursions(abs_gap, threshold=20.0, revert_level=10.0, bar_minutes=5.0)
        assert r["n_excursions"] == 0
        assert r["frac_bars_in_excursion"] == 0.0


# ============================================================
# Structural premium vs transient
# ============================================================

class TestStructuralPremium:

    def test_constant_premium_is_structural_not_transient(self):
        # A constant +50 bps gap: sign-stable structural premium, ZERO transient
        # (mean-removed) excursions, but raw excursions cover ~all bars.
        k, o = pair_from_gap([50.0] * 300)
        cfg = ds.DislocationConfig(cost_grid_bps=(20,), round_trip_cost_bps=20.0)
        a = ds.analyze_dislocation(k, o, cfg)

        assert a["structural"]["mean_gap_bps"] == pytest.approx(50.0)
        assert a["structural"]["sign_stability"] == pytest.approx(1.0)
        # Transient (mean-removed) gap is ~0 everywhere -> no excursions over 20.
        assert a["excursions"]["demeaned"][20]["n_excursions"] == 0
        # Raw gap is 50 > 20 everywhere -> one big sticky excursion covering all bars.
        assert a["excursions"]["raw"][20]["frac_bars_in_excursion"] == pytest.approx(1.0)

    def test_transient_gap_registers_demeaned_excursions(self):
        # Gap spikes to 40 bps periodically around mean ~0 -> transient excursions.
        pattern = ([40.0, 0.0, 0.0, 0.0] * 50)  # mean 10; demeaned spikes = 30 > 20
        k, o = pair_from_gap(pattern)
        cfg = ds.DislocationConfig(cost_grid_bps=(20,), round_trip_cost_bps=20.0)
        a = ds.analyze_dislocation(k, o, cfg)
        assert a["excursions"]["demeaned"][20]["n_excursions"] > 0


# ============================================================
# Mean-reversion half-life
# ============================================================

class TestHalfLife:

    def test_ar1_half_life_recovered(self):
        rng = np.random.default_rng(0)
        phi = 0.5
        n = 20000
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.normal(0, 1)
        ac, hl = ds._half_life(x)
        assert ac == pytest.approx(0.5, abs=0.05)
        # half-life = -ln2/ln(0.5) = 1.0 bar exactly for phi=0.5.
        assert hl == pytest.approx(1.0, abs=0.2)

    def test_flat_series_returns_nan(self):
        ac, hl = ds._half_life(np.full(100, 5.0))
        assert np.isnan(ac) and np.isnan(hl)


# ============================================================
# Scanner table
# ============================================================

class TestScanBasket:

    def test_table_shape_and_sort(self):
        # ALT has transient excursions; CTRL is a constant premium (none).
        alt = pair_from_gap([40.0, 0.0, 0.0, 0.0] * 60)
        ctrl = pair_from_gap([50.0] * 240)
        basket = {"CTRL": ctrl, "ALT": alt}
        cfg = ds.DislocationConfig(cost_grid_bps=(10, 20, 30), round_trip_cost_bps=20.0)
        liquidity = {"ALT": 1e8, "CTRL": 5e9}

        df = ds.scan_basket(basket, cfg, liquidity)
        assert len(df) == 2
        assert list(df["asset"]) == ["ALT", "CTRL"]  # ALT ranks first (more transient)
        for col in ("asset", "liquidity_usd_24h", "mean_abs_gap_bps", "p95_abs_gap_bps",
                    "struct_mean_gap_bps", "sign_stability", "half_life_min",
                    "transient_exc_per_day@20"):
            assert col in df.columns
        assert df.loc[df["asset"] == "CTRL", "transient_exc_per_day@20"].iloc[0] == 0.0

    def test_liquidity_defaults_to_nan(self):
        basket = {"X": pair_from_gap([40.0, 0.0] * 100)}
        df = ds.scan_basket(basket, ds.DislocationConfig())
        assert np.isnan(df.loc[0, "liquidity_usd_24h"])
