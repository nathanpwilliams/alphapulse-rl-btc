"""
tests/test_regime_backtest.py - Unit tests for
src/strategies/regime/backtest.py

Synthetic data, no network. The non-negotiable guards are look-ahead and
detection lag: a regime label used to decide bar t must depend only on data <= t,
and the position held over bar t must come from the regime known through t-1.
Everything else (cost accounting, buy-and-hold benchmark, metrics, favorability,
walk-forward cadence, auto state selection) is also covered.

Run:
    pytest tests/test_regime_backtest.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.regime import backtest as bt  # noqa: E402


# ============================================================
# Synthetic builders
# ============================================================

def synth_features(n: int, seed: int = 0, block: int = 30) -> pd.DataFrame:
    """Two clearly-separated regimes (calm/uptrend vs high-vol/choppy) in blocks.

    Deliberately bimodal so a 2-state GaussianHMM fits stably and deterministically.
    """
    rng = np.random.default_rng(seed)
    lr = np.empty(n)
    vol = np.empty(n)
    tr = np.empty(n)
    for b in range(0, n, block):
        sl = slice(b, min(b + block, n))
        size = min(b + block, n) - b
        high_vol = (b // block) % 2 == 1
        if high_vol:
            lr[sl] = rng.normal(0.0, 0.03, size)
            vol[sl] = np.abs(rng.normal(0.03, 0.003, size))
            tr[sl] = rng.normal(-0.01, 0.005, size)
        else:
            lr[sl] = rng.normal(0.0015, 0.005, size)
            vol[sl] = np.abs(rng.normal(0.006, 0.001, size))
            tr[sl] = rng.normal(0.01, 0.005, size)
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"log_return": lr, "realized_vol": vol, "trend": tr}, index=idx)


def synth_ohlcv(n: int, seed: int = 0, block: int = 30) -> pd.DataFrame:
    """OHLCV (close) with alternating calm-drift / high-vol blocks."""
    rng = np.random.default_rng(seed)
    rets = np.empty(n)
    for b in range(0, n, block):
        sl = slice(b, min(b + block, n))
        size = min(b + block, n) - b
        if (b // block) % 2 == 1:
            rets[sl] = rng.normal(0.0, 0.03, size)
        else:
            rets[sl] = rng.normal(0.0015, 0.005, size)
    close = 40000.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"close": close}, index=idx)


def fast_config(**kw) -> bt.RegimeBacktestConfig:
    base = dict(n_states=2, refit_every=20, min_train_bars=60,
                n_restarts=2, n_iter=50, n_states_grid=(2, 3))
    base.update(kw)
    return bt.RegimeBacktestConfig(**base)


# ============================================================
# LOOK-AHEAD GUARD (the most important test in the suite)
# ============================================================

class TestNoLookAhead:

    @pytest.mark.parametrize("offset", [2, 30])
    def test_regime_label_depends_only_on_past(self, offset):
        """Modifying data AFTER bar t must not change bar t's regime label.

        offset=2 perturbs a bar in the same refit block but after t; offset=30
        perturbs a bar in a later block. Neither may alter the label at t.
        """
        feats = synth_features(160, seed=1)
        cfg = fast_config()
        base = bt.assign_regimes_walk_forward(feats, cfg)["labels"]

        t_pos = 70  # a labeled bar (>= min_train_bars=60)
        t_ts = feats.index[t_pos]

        perturbed = feats.copy()
        p = t_pos + offset
        # Blow up a later feature row; a look-ahead bug would leak this into t.
        perturbed.iloc[p] = perturbed.iloc[p] * 10.0 + 5.0
        after = bt.assign_regimes_walk_forward(perturbed, cfg)["labels"]

        # Labels up to and including t must be byte-identical.
        pd.testing.assert_frame_equal(base.loc[:t_ts], after.loc[:t_ts])

    def test_scaler_fit_on_train_only(self):
        # A future outlier must not shift the standardization of past labels either
        # (covered by the parametrized test, but assert the first labeled bar too).
        feats = synth_features(160, seed=2)
        cfg = fast_config()
        base = bt.assign_regimes_walk_forward(feats, cfg)["labels"]
        perturbed = feats.copy()
        perturbed.iloc[-1] = perturbed.iloc[-1] * 100.0
        after = bt.assign_regimes_walk_forward(perturbed, cfg)["labels"]
        first_ts = base.index[0]
        assert base.loc[first_ts, "regime"] == after.loc[first_ts, "regime"]


# ============================================================
# DETECTION LAG GUARD
# ============================================================

class TestDetectionLag:

    def test_position_uses_prior_bar_regime(self):
        fav = pd.Series([True, True, False, False, True],
                        index=pd.date_range("2023-01-01", periods=5, freq="4h", tz="UTC"))
        pos = bt._positions_from_labels(fav)
        # position[t] = favorable[t-1]; first bar flat (no prior).
        assert list(pos.to_numpy()) == [0.0, 1.0, 1.0, 0.0, 0.0]

    def test_lag_end_to_end(self):
        ohlcv = synth_ohlcv(200, seed=3)
        cfg = fast_config(refit_every=30, transaction_cost_bps=0.0)
        result = bt.run_backtest(ohlcv, cfg)
        frame = result["frame"]
        # The acted position must equal the favorable flag shifted by one bar.
        expected = frame["favorable"].shift(1, fill_value=False).astype(float)
        # First row has no prior label within the traded window; compare from row 1.
        assert np.allclose(frame["position"].to_numpy()[1:], expected.to_numpy()[1:])


# ============================================================
# TRANSACTION COSTS
# ============================================================

class TestTransactionCosts:

    def test_n_flips_incur_n_costs(self):
        idx = pd.date_range("2023-01-01", periods=5, freq="4h", tz="UTC")
        positions = pd.Series([0.0, 1.0, 1.0, 0.0, 1.0], index=idx)
        returns = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=idx)
        sim = bt._simulate(positions, returns, cost_bps=10.0)
        # Trades at t1 (0->1), t3 (1->0), t4 (0->1) = 3; each costs 10bps = 0.001.
        assert sim["n_trades"] == 3
        assert sim["equity"].iloc[-1] == pytest.approx((1.0 - 0.001) ** 3)

    def test_no_change_no_cost(self):
        idx = pd.date_range("2023-01-01", periods=4, freq="4h", tz="UTC")
        positions = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx)
        returns = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)
        sim = bt._simulate(positions, returns, cost_bps=25.0)
        # Only the initial entry (0->1) trades; no cost thereafter.
        assert sim["n_trades"] == 1
        assert sim["equity"].iloc[-1] == pytest.approx(1.0 - 0.0025)


# ============================================================
# BUY-AND-HOLD BENCHMARK
# ============================================================

class TestBuyAndHold:

    def test_buy_and_hold_matches_cumprod(self):
        idx = pd.date_range("2023-01-01", periods=3, freq="4h", tz="UTC")
        positions = pd.Series([1.0, 1.0, 1.0], index=idx)
        returns = pd.Series([0.05, -0.02, 0.03], index=idx)
        sim = bt._simulate(positions, returns, cost_bps=0.0)  # no cost -> pure cumprod
        expected = np.cumprod(1.0 + returns.to_numpy())
        assert np.allclose(sim["equity"].to_numpy(), expected)


# ============================================================
# FAVORABILITY RULE
# ============================================================

class TestFavorability:

    def _char(self):
        return {"regimes": {
            0: {"return_profile": {"mean_log_return": 0.001},
                "volatility_profile": {"mean_realized_vol": 0.005}},
            1: {"return_profile": {"mean_log_return": -0.001},
                "volatility_profile": {"mean_realized_vol": 0.030}},
            2: {"return_profile": {"mean_log_return": 0.002},
                "volatility_profile": {"mean_realized_vol": 0.020}},
        }}

    def test_positive_return_below_median_vol(self):
        fav = bt._favorability_map(self._char(), "positive_return_below_median_vol")
        # vols [0.005, 0.03, 0.02] -> median 0.02.
        # s0: +ret & 0.005<=0.02 -> True; s1: -ret -> False; s2: +ret & 0.02<=0.02 -> True
        assert fav == {0: True, 1: False, 2: True}

    def test_positive_return_rule(self):
        fav = bt._favorability_map(self._char(), "positive_return")
        assert fav == {0: True, 1: False, 2: True}

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="Unknown favorable_regime_rule"):
            bt._favorability_map(self._char(), "nonexistent")


# ============================================================
# METRICS
# ============================================================

class TestMetrics:

    def test_max_drawdown(self):
        eq = pd.Series([1.0, 1.2, 0.9, 1.0])
        # running max 1.2; trough 0.9 -> (0.9/1.2 - 1) = -0.25.
        assert bt._max_drawdown(eq) == pytest.approx(-0.25)

    def test_sharpe_and_returns_hand_computed(self):
        idx = pd.date_range("2023-01-01", periods=3, freq="4h", tz="UTC")
        rets = pd.Series([0.01, -0.01, 0.02], index=idx)
        equity = pd.Series(np.cumprod(1.0 + rets.to_numpy()), index=idx)
        ppy = 8760 / 4.0
        m = bt._compute_metrics(rets, equity, ppy)

        r = rets.to_numpy()
        assert m["sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(ppy))
        assert m["total_return"] == pytest.approx(equity.iloc[-1] - 1.0)
        assert m["annualized_return"] == pytest.approx(
            (1.0 + (equity.iloc[-1] - 1.0)) ** (ppy / 3) - 1.0
        )
        assert m["annualized_vol"] == pytest.approx(r.std() * np.sqrt(ppy))

    def test_sharpe_nan_on_zero_vol(self):
        idx = pd.date_range("2023-01-01", periods=3, freq="4h", tz="UTC")
        rets = pd.Series([0.0, 0.0, 0.0], index=idx)
        equity = pd.Series([1.0, 1.0, 1.0], index=idx)
        m = bt._compute_metrics(rets, equity, 2190.0)
        assert np.isnan(m["sharpe"])


# ============================================================
# WALK-FORWARD MECHANICS
# ============================================================

class TestWalkForward:

    def test_refit_cadence_count(self):
        feats = synth_features(140, seed=4)
        cfg = fast_config(min_train_bars=60, refit_every=20)  # blocks at 60,80,100,120
        wf = bt.assign_regimes_walk_forward(feats, cfg)
        assert wf["n_refits"] == 4  # ceil((140-60)/20)

    def test_labels_start_at_min_train_bars(self):
        feats = synth_features(140, seed=5)
        cfg = fast_config(min_train_bars=60, refit_every=20)
        wf = bt.assign_regimes_walk_forward(feats, cfg)
        assert wf["labels"].index[0] == feats.index[60]
        assert len(wf["labels"]) == 140 - 60

    def test_too_little_history_raises(self):
        feats = synth_features(40, seed=6)
        cfg = fast_config(min_train_bars=60)
        with pytest.raises(ValueError, match="Not enough feature rows"):
            bt.assign_regimes_walk_forward(feats, cfg)

    def test_auto_n_states_selection_runs(self):
        feats = synth_features(140, seed=7)
        cfg = fast_config(n_states=None, min_train_bars=60, refit_every=40,
                          n_states_grid=(2, 3))
        wf = bt.assign_regimes_walk_forward(feats, cfg)
        assert wf["n_states"] in (2, 3)
        assert not wf["labels"].empty


# ============================================================
# END-TO-END
# ============================================================

class TestRunBacktest:

    def test_returns_expected_structure(self):
        ohlcv = synth_ohlcv(200, seed=8)
        cfg = fast_config(refit_every=40)
        result = bt.run_backtest(ohlcv, cfg)

        assert set(result.keys()) >= {"frame", "metrics", "summary", "n_refits"}
        assert set(result["metrics"].keys()) == {"filtered", "buy_and_hold"}
        s = result["summary"]
        for k in ("symbol", "resolution", "n_states_selected", "current_regime",
                  "current_favorable", "time_in_market", "n_trades",
                  "sharpe_filtered", "sharpe_bh", "sharpe_uplift",
                  "max_drawdown_filtered", "max_drawdown_bh"):
            assert k in s
        assert 0.0 <= s["time_in_market"] <= 1.0

    def test_report_renders(self):
        ohlcv = synth_ohlcv(200, seed=9)
        cfg = fast_config(refit_every=40)
        result = bt.run_backtest(ohlcv, cfg)
        text = bt.format_report(result, cfg)
        assert "REGIME-AS-FILTER BACKTEST" in text
        assert "Buy & hold" in text
