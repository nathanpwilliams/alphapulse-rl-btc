"""
tests/test_funding_analyzer.py - Unit tests for
src/strategies/funding/analyzer.py

All synthetic, no network. Funding DataFrames are constructed in-test so we can
hand-compute the expected statistics and pin down the unit conversions
(per-hour rate -> bps -> annualized), which are the analyzer's main bug surface.

Run:
    pytest tests/test_funding_analyzer.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.funding.analyzer import (  # noqa: E402
    HOURS_PER_YEAR,
    MAX_POSITIVE_HOLD_HOURS,
    FundingAnalyzerConfig,
    analyze_funding,
    format_summary,
    format_sweep,
    sweep_costs,
)


# ============================================================
# Synthetic funding builders
# ============================================================

def make_funding(rates, start: str = "2024-01-01") -> pd.DataFrame:
    """Build an hourly-cadence funding DataFrame from a sequence of rates."""
    rates = np.asarray(rates, dtype=float)
    idx = pd.date_range(start, periods=len(rates), freq="1h", tz="UTC")
    return pd.DataFrame({"funding_rate": rates}, index=idx)


def constant_funding(rate: float, n: int = 1000) -> pd.DataFrame:
    return make_funding(np.full(n, rate))


def alternating_funding(mag: float = 1e-4, n: int = 1000) -> pd.DataFrame:
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return make_funding(signs * mag)


# ============================================================
# Keys / structure
# ============================================================

class TestStructure:

    def test_returns_all_expected_keys(self):
        stats = analyze_funding(constant_funding(1e-5), FundingAnalyzerConfig())
        expected = {
            "coverage", "sign_distribution", "magnitude", "gross_yield",
            "cost_hurdle", "sign_flips", "net_by_holding_period", "config",
        }
        assert set(stats.keys()) == expected

    def test_net_table_has_all_holding_periods(self):
        cfg = FundingAnalyzerConfig()
        stats = analyze_funding(constant_funding(1e-5), cfg)
        assert set(stats["net_by_holding_period"].keys()) == set(cfg.holding_periods_hours)
        for row in stats["net_by_holding_period"].values():
            assert set(row.keys()) == {"gross_bps", "net_bps", "annualized_net_pct"}

    def test_rejects_missing_column(self):
        df = pd.DataFrame({"wrong": [1.0]}, index=pd.date_range("2024-01-01", periods=1, tz="UTC"))
        with pytest.raises(ValueError, match="funding_rate"):
            analyze_funding(df, FundingAnalyzerConfig())

    def test_rejects_empty(self):
        df = pd.DataFrame({"funding_rate": []}, index=pd.DatetimeIndex([], tz="UTC"))
        with pytest.raises(ValueError, match="empty"):
            analyze_funding(df, FundingAnalyzerConfig())

    def test_summary_renders(self):
        stats = analyze_funding(constant_funding(1e-5), FundingAnalyzerConfig())
        text = format_summary(stats)
        assert "FUNDING-RATE ARBITRAGE ANALYZER" in text
        assert "Net-after-cost" in text


# ============================================================
# Constant positive funding: sign, annualized yield, break-even
# ============================================================

class TestConstantPositive:

    RATE = 1e-5  # per-hour proportional rate

    def test_sign_distribution_all_positive(self):
        stats = analyze_funding(constant_funding(self.RATE), FundingAnalyzerConfig())
        sd = stats["sign_distribution"]
        assert sd["frac_positive"] == 1.0
        assert sd["frac_negative"] == 0.0
        assert sd["frac_zero"] == 0.0

    def test_annualized_yield_matches_hand_computed(self):
        stats = analyze_funding(constant_funding(self.RATE), FundingAnalyzerConfig())
        # rate * 8760 * 100 percent
        expected = self.RATE * HOURS_PER_YEAR * 100
        assert stats["magnitude"]["mean_annualized_pct"] == pytest.approx(expected)
        assert stats["gross_yield"]["annualized_gross_pct"] == pytest.approx(expected)

    def test_break_even_hours_matches_formula(self):
        cfg = FundingAnalyzerConfig(round_trip_cost_bps=20.0)
        stats = analyze_funding(constant_funding(self.RATE), cfg)
        # break_even = cost_bps / (rate * 10_000)
        expected = 20.0 / (self.RATE * 10_000)
        assert stats["cost_hurdle"]["break_even_hours"] == pytest.approx(expected)
        assert stats["cost_hurdle"]["break_even_days"] == pytest.approx(expected / 24.0)

    def test_zero_funding_break_even_infinite(self):
        stats = analyze_funding(constant_funding(0.0), FundingAnalyzerConfig())
        assert stats["cost_hurdle"]["break_even_hours"] == float("inf")


# ============================================================
# Sign flips
# ============================================================

class TestSignFlips:

    def test_alternating_flip_fraction_is_one(self):
        stats = analyze_funding(alternating_funding(1e-4, n=1000), FundingAnalyzerConfig())
        # Every consecutive pair changes sign -> fraction ~ 1.0
        assert stats["sign_flips"]["flip_fraction"] == pytest.approx(1.0)
        # Hourly cadence => one hour between flips.
        assert stats["sign_flips"]["mean_hours_between_flips"] == pytest.approx(1.0)

    def test_constant_has_no_flips(self):
        stats = analyze_funding(constant_funding(1e-5), FundingAnalyzerConfig())
        assert stats["sign_flips"]["n_flips"] == 0
        assert stats["sign_flips"]["flip_fraction"] == 0.0
        assert stats["sign_flips"]["mean_hours_between_flips"] == float("inf")


# ============================================================
# Net-after-cost accounting
# ============================================================

class TestNetAfterCost:

    def test_net_bps_matches_hand_computed(self):
        rate = 1e-4  # 1 bps/hr
        cost = 20.0
        cfg = FundingAnalyzerConfig(round_trip_cost_bps=cost)
        stats = analyze_funding(constant_funding(rate), cfg)

        h = 168  # 7 days
        # gross = rate * 10_000 * H = 1 bps/hr * 168 hr = 168 bps
        expected_gross = rate * 10_000 * h
        expected_net = expected_gross - cost
        row = stats["net_by_holding_period"][h]
        assert row["gross_bps"] == pytest.approx(expected_gross)
        assert row["net_bps"] == pytest.approx(expected_net)

    def test_annualized_net_matches_hand_computed(self):
        rate = 1e-4
        cost = 20.0
        cfg = FundingAnalyzerConfig(round_trip_cost_bps=cost)
        stats = analyze_funding(constant_funding(rate), cfg)

        h = 720
        gross = rate * 10_000 * h
        net = gross - cost
        cycles = HOURS_PER_YEAR / h
        expected_ann = (net / 10_000) * cycles * 100
        assert stats["net_by_holding_period"][h]["annualized_net_pct"] == pytest.approx(expected_ann)

    def test_short_hold_can_be_net_negative(self):
        # Tiny funding, real cost: 24h hold should not clear the hurdle.
        cfg = FundingAnalyzerConfig(round_trip_cost_bps=20.0)
        stats = analyze_funding(constant_funding(1e-6), cfg)
        assert stats["net_by_holding_period"][24]["net_bps"] < 0


# ============================================================
# Cost parameter flow-through
# ============================================================

class TestCostParameter:

    def test_doubling_cost_doubles_break_even(self):
        rate = 1e-5
        s1 = analyze_funding(constant_funding(rate), FundingAnalyzerConfig(round_trip_cost_bps=10.0))
        s2 = analyze_funding(constant_funding(rate), FundingAnalyzerConfig(round_trip_cost_bps=20.0))
        be1 = s1["cost_hurdle"]["break_even_hours"]
        be2 = s2["cost_hurdle"]["break_even_hours"]
        assert be2 == pytest.approx(2.0 * be1)


# ============================================================
# Unit-conversion sanity
# ============================================================

class TestUnitConversions:

    def test_hours_per_year_constant(self):
        assert HOURS_PER_YEAR == 8760

    def test_known_rate_annualizes_correctly(self):
        rate = 2.5e-5
        stats = analyze_funding(constant_funding(rate), FundingAnalyzerConfig())
        # rate * 8760 -> proportional annual; * 100 -> percent
        assert stats["magnitude"]["mean_annualized_pct"] == pytest.approx(rate * 8760 * 100)

    def test_coverage_hours_span(self):
        # 100 hourly obs -> 99 hours between first and last timestamp.
        stats = analyze_funding(constant_funding(1e-5, n=100), FundingAnalyzerConfig())
        assert stats["coverage"]["n_observations"] == 100
        assert stats["coverage"]["total_hours_covered"] == pytest.approx(99.0)


# ============================================================
# Cost-sensitivity sweep
# ============================================================

class TestSweepCosts:

    GRID = [5.0, 10.0, 15.0, 20.0, 25.0]

    def test_one_entry_per_cost_level(self):
        sweep = sweep_costs(constant_funding(1e-5), FundingAnalyzerConfig(), self.GRID)
        assert set(sweep.keys()) == set(self.GRID)
        for cost, entry in sweep.items():
            assert entry["round_trip_cost_bps"] == cost
            assert set(entry["net_by_holding_period"].keys()) == set(
                FundingAnalyzerConfig().holding_periods_hours
            )

    def test_break_even_scales_linearly_with_cost(self):
        rate = 1e-5
        sweep = sweep_costs(constant_funding(rate), FundingAnalyzerConfig(), [10.0, 20.0])
        be10 = sweep[10.0]["break_even_hours"]
        be20 = sweep[20.0]["break_even_hours"]
        assert be20 == pytest.approx(2.0 * be10)
        # And matches the closed form for the 10 bps level.
        assert be10 == pytest.approx(10.0 / (rate * 10_000))

    def test_net_bps_matches_hand_computed(self):
        rate = 1e-4  # 1 bps/hr
        cost = 15.0
        sweep = sweep_costs(constant_funding(rate), FundingAnalyzerConfig(), [cost])
        h = 336  # 14 days
        expected_gross = rate * 10_000 * h  # 336 bps
        expected_net = expected_gross - cost
        row = sweep[cost]["net_by_holding_period"][h]
        assert row["gross_bps"] == pytest.approx(expected_gross)
        assert row["net_bps"] == pytest.approx(expected_net)

    def test_break_even_monotonic_in_cost(self):
        sweep = sweep_costs(constant_funding(1e-5), FundingAnalyzerConfig(), self.GRID)
        be = [sweep[c]["break_even_hours"] for c in sorted(sweep)]
        # Higher cost => strictly later break-even.
        assert all(be[i] < be[i + 1] for i in range(len(be) - 1))

    def test_min_positive_hold_equals_break_even(self):
        sweep = sweep_costs(constant_funding(1e-5), FundingAnalyzerConfig(), [20.0])
        entry = sweep[20.0]
        assert entry["min_positive_hold_hours"] == pytest.approx(entry["break_even_hours"])

    def test_min_positive_hold_none_beyond_cap(self):
        # Tiny funding + high cost pushes break-even past the 90-day cap.
        sweep = sweep_costs(constant_funding(1e-7), FundingAnalyzerConfig(), [20.0])
        entry = sweep[20.0]
        assert entry["break_even_hours"] > MAX_POSITIVE_HOLD_HOURS
        assert entry["min_positive_hold_hours"] is None
        assert entry["min_positive_hold_days"] is None

    def test_empty_grid_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            sweep_costs(constant_funding(1e-5), FundingAnalyzerConfig(), [])

    def test_sweep_summary_renders(self):
        sweep = sweep_costs(constant_funding(1e-5), FundingAnalyzerConfig(), self.GRID)
        text = format_sweep(sweep, "PF_XBTUSD")
        assert "COST-SENSITIVITY SWEEP" in text
        assert "Summary" in text


class TestSweepMatchesSingleCost:
    """The refactor must not create two divergent funding-math paths."""

    def test_sweep_and_analyze_agree_on_net(self):
        rate = 3.3e-5
        cost = 12.0
        cfg = FundingAnalyzerConfig(round_trip_cost_bps=cost)
        df = constant_funding(rate)

        single = analyze_funding(df, cfg)["net_by_holding_period"]
        swept = sweep_costs(df, cfg, [cost])[cost]["net_by_holding_period"]

        assert single.keys() == swept.keys()
        for h in single:
            assert single[h]["gross_bps"] == pytest.approx(swept[h]["gross_bps"])
            assert single[h]["net_bps"] == pytest.approx(swept[h]["net_bps"])
            assert single[h]["annualized_net_pct"] == pytest.approx(
                swept[h]["annualized_net_pct"]
            )

    def test_sweep_and_analyze_agree_on_break_even(self):
        rate = 3.3e-5
        cost = 12.0
        cfg = FundingAnalyzerConfig(round_trip_cost_bps=cost)
        df = constant_funding(rate)
        single_be = analyze_funding(df, cfg)["cost_hurdle"]["break_even_hours"]
        swept_be = sweep_costs(df, cfg, [cost])[cost]["break_even_hours"]
        assert single_be == pytest.approx(swept_be)
