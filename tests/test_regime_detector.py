"""
tests/test_regime_detector.py - Unit tests for
src/strategies/regime/detector.py

Synthetic data, no network. The properties that matter for a regime diagnostic:
features are computed correctly and without look-ahead, the HMM recovers
deliberately-distinct synthetic regimes, persistence/characterization math is
correct, fits are deterministic under a fixed seed, and AIC/BIC are computed
correctly from log-likelihood and parameter count.

Run:
    pytest tests/test_regime_detector.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.regime import detector as rd  # noqa: E402


# ============================================================
# Synthetic builders
# ============================================================

def make_ohlcv_from_returns(returns: np.ndarray, start_price: float = 40000.0,
                            start: str = "2023-01-01") -> pd.DataFrame:
    """Build a minimal OHLCV frame (close-only sufficient) from a return series."""
    close = start_price * np.exp(np.cumsum(returns))
    idx = pd.date_range(start, periods=len(close), freq="4h", tz="UTC")
    return pd.DataFrame({"close": close}, index=idx)


def two_regime_data(seed: int = 0, block: int = 120, n_blocks: int = 6):
    """Alternating low-vol/drift and high-vol/choppy blocks.

    Returns (ohlcv, ground_truth_series) where ground_truth is 0 for the low-vol
    regime and 1 for the high-vol regime, indexed like ohlcv.
    """
    rng = np.random.default_rng(seed)
    rets = []
    truth = []
    for b in range(n_blocks):
        if b % 2 == 0:
            rets.append(rng.normal(0.0015, 0.006, block))   # low vol, mild uptrend
            truth.append(np.zeros(block, dtype=int))
        else:
            rets.append(rng.normal(0.0, 0.035, block))       # high vol, choppy
            truth.append(np.ones(block, dtype=int))
    rets = np.concatenate(rets)
    truth = np.concatenate(truth)
    ohlcv = make_ohlcv_from_returns(rets)
    truth_s = pd.Series(truth, index=ohlcv.index)
    return ohlcv, truth_s


def fast_config(**kw) -> rd.RegimeConfig:
    """RegimeConfig with restarts/iters trimmed so tests stay quick."""
    base = dict(n_restarts=3, n_iter=100, n_states_grid=(2, 3))
    base.update(kw)
    return rd.RegimeConfig(**base)


# ============================================================
# Feature engineering
# ============================================================

class TestFeatures:

    def test_log_return_correct(self):
        close = np.array([100.0, 101.0, 103.0, 102.0, 104.0, 105.0])
        idx = pd.date_range("2023-01-01", periods=len(close), freq="4h", tz="UTC")
        ohlcv = pd.DataFrame({"close": close}, index=idx)
        cfg = rd.RegimeConfig(vol_window=2, trend_window=2)
        feats = rd.compute_features(ohlcv, cfg)
        # Recompute log returns independently and compare on the surviving rows.
        full_lr = np.log(pd.Series(close, index=idx) / pd.Series(close, index=idx).shift(1))
        assert np.allclose(feats["log_return"].to_numpy(), full_lr.loc[feats.index].to_numpy())

    def test_realized_vol_matches_rolling_std(self):
        rng = np.random.default_rng(1)
        ohlcv = make_ohlcv_from_returns(rng.normal(0, 0.01, 200))
        cfg = rd.RegimeConfig(vol_window=8, trend_window=12)
        feats = rd.compute_features(ohlcv, cfg)
        lr = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
        expected_vol = lr.rolling(8).std().loc[feats.index]
        assert np.allclose(feats["realized_vol"].to_numpy(), expected_vol.to_numpy())

    def test_trend_matches_sma_distance(self):
        rng = np.random.default_rng(2)
        ohlcv = make_ohlcv_from_returns(rng.normal(0, 0.01, 200))
        cfg = rd.RegimeConfig(vol_window=8, trend_window=12)
        feats = rd.compute_features(ohlcv, cfg)
        close = ohlcv["close"]
        sma = close.rolling(12).mean()
        expected_trend = ((close - sma) / sma).loc[feats.index]
        assert np.allclose(feats["trend"].to_numpy(), expected_trend.to_numpy())

    def test_no_lookahead(self):
        rng = np.random.default_rng(3)
        ohlcv = make_ohlcv_from_returns(rng.normal(0, 0.01, 100))
        cfg = rd.RegimeConfig(vol_window=8, trend_window=12)
        feats_before = rd.compute_features(ohlcv, cfg)

        # Perturb ONLY the last bar's close; earlier feature rows must not change.
        ohlcv2 = ohlcv.copy()
        ohlcv2.iloc[-1, ohlcv2.columns.get_loc("close")] *= 1.5
        feats_after = rd.compute_features(ohlcv2, cfg)

        common = feats_before.index[:-1]  # everything except the perturbed last bar
        pd.testing.assert_frame_equal(
            feats_before.loc[common], feats_after.loc[common]
        )

    def test_warmup_rows_dropped(self):
        rng = np.random.default_rng(4)
        n = 200
        ohlcv = make_ohlcv_from_returns(rng.normal(0, 0.01, n))
        cfg = rd.RegimeConfig(vol_window=8, trend_window=12)
        feats = rd.compute_features(ohlcv, cfg)
        # First valid row = max(vol valid @8, trend valid @11) = index 11.
        assert feats.index[0] == ohlcv.index[11]
        assert len(feats) == n - 11
        assert not feats.isna().any().any()


# ============================================================
# Information criteria
# ============================================================

class TestInfoCriteria:

    def test_n_params_full_cov_hand_computed(self):
        # 2 states, 3 features, full cov:
        #   start=1, trans=2, means=6, cov=2*(3*4/2)=12 -> 21
        assert rd._hmm_n_params(2, 3, "full") == 21
        # 3 states, 3 features, full cov:
        #   start=2, trans=6, means=9, cov=3*6=18 -> 35
        assert rd._hmm_n_params(3, 3, "full") == 35

    def test_aic_bic_hand_computed(self):
        ll, p, n = -100.0, 21, 1000
        aic, bic = rd._aic_bic(ll, p, n)
        assert aic == pytest.approx(-2 * ll + 2 * p)          # 242
        assert bic == pytest.approx(-2 * ll + p * np.log(n))  # 200 + 21*ln(1000)


# ============================================================
# HMM fitting / recovery
# ============================================================

class TestHMMRecovery:

    def test_recovers_two_synthetic_regimes(self):
        from sklearn.metrics import adjusted_rand_score

        ohlcv, truth = two_regime_data(seed=0)
        cfg = fast_config()
        feats = rd.compute_features(ohlcv, cfg)
        fit = rd.fit_regimes(feats, n_states=2, config=cfg)

        truth_aligned = truth.loc[feats.index].to_numpy()
        # ARI is invariant to label permutation (HMM labels are arbitrary).
        ari = adjusted_rand_score(truth_aligned, fit["states"])
        assert ari > 0.3, f"HMM failed to recover the two regimes (ARI={ari:.3f})"

    def test_determinism_same_seed(self):
        ohlcv, _ = two_regime_data(seed=0)
        cfg = fast_config()
        feats = rd.compute_features(ohlcv, cfg)
        fit1 = rd.fit_regimes(feats, n_states=2, config=cfg)
        fit2 = rd.fit_regimes(feats, n_states=2, config=cfg)
        assert fit1["log_likelihood"] == pytest.approx(fit2["log_likelihood"])
        assert np.array_equal(fit1["states"], fit2["states"])

    def test_fit_returns_expected_keys(self):
        ohlcv, _ = two_regime_data(seed=1)
        cfg = fast_config()
        feats = rd.compute_features(ohlcv, cfg)
        fit = rd.fit_regimes(feats, n_states=2, config=cfg)
        for k in ("model", "states", "index", "log_likelihood", "n_params",
                  "aic", "bic", "n_states", "converged"):
            assert k in fit
        assert len(fit["states"]) == len(feats)

    def test_select_n_states_reports_all(self):
        ohlcv, _ = two_regime_data(seed=2)
        cfg = fast_config(n_states_grid=(2, 3))
        feats = rd.compute_features(ohlcv, cfg)
        sel = rd.select_n_states(feats, cfg)
        assert set(sel["per_n_states"].keys()) == {2, 3}
        assert sel["aic_preferred"] in (2, 3)
        assert sel["bic_preferred"] in (2, 3)
        for row in sel["per_n_states"].values():
            assert {"log_likelihood", "n_params", "aic", "bic", "converged"} <= set(row)


# ============================================================
# Characterization
# ============================================================

class TestCharacterization:

    def _dummy_features(self, n: int, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
        return pd.DataFrame(
            {"log_return": rng.normal(0, 0.01, n),
             "realized_vol": np.abs(rng.normal(0.01, 0.002, n)),
             "trend": rng.normal(0, 0.01, n)},
            index=idx,
        )

    def test_persistence_mean_duration_correct(self):
        # State 0 runs of length 3 and 4 (mean 3.5); state 1 a single run of 2.
        seq = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0])
        feats = self._dummy_features(len(seq))
        ohlcv = feats  # unused beyond signature
        cfg = rd.RegimeConfig(resolution="4h")
        char = rd.characterize_regimes(feats, seq, ohlcv, cfg)
        assert char["regimes"][0]["persistence"]["mean_bars"] == pytest.approx(3.5)
        assert char["regimes"][1]["persistence"]["mean_bars"] == pytest.approx(2.0)
        # 4H bars -> hours = bars * 4.
        assert char["regimes"][0]["persistence"]["mean_hours"] == pytest.approx(3.5 * 4)
        assert char["regimes"][0]["persistence"]["max_bars"] == pytest.approx(4.0)

    def test_frequency_sums_to_one(self):
        seq = np.array([0, 0, 1, 1, 1, 2, 0, 1])
        feats = self._dummy_features(len(seq))
        char = rd.characterize_regimes(feats, seq, feats, rd.RegimeConfig())
        total = sum(char["regimes"][s]["frequency"] for s in char["regimes"])
        assert total == pytest.approx(1.0)

    def test_transition_matrix_rows_sum_to_one(self):
        seq = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0])
        feats = self._dummy_features(len(seq))
        char = rd.characterize_regimes(feats, seq, feats, rd.RegimeConfig())
        tmat = np.array(char["transition_matrix"])
        assert np.allclose(tmat.sum(axis=1), 1.0)
        assert char["transition_matrix_source"] == "empirical"

    def test_characterization_output_shape(self):
        seq = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        feats = self._dummy_features(len(seq))
        char = rd.characterize_regimes(feats, seq, feats, rd.RegimeConfig())
        for s in char["regimes"]:
            r = char["regimes"][s]
            assert {"label", "frequency", "n_bars", "persistence", "return_profile",
                    "volatility_profile", "directionality"} <= set(r)
            assert {"mean_bars", "median_bars", "max_bars", "mean_hours",
                    "mean_days", "n_episodes"} <= set(r["persistence"])
        assert {"return_spread", "vol_spread"} <= set(char["distinctness"])

    def test_length_mismatch_raises(self):
        feats = self._dummy_features(10)
        with pytest.raises(ValueError, match="equal length"):
            rd.characterize_regimes(feats, np.array([0, 1, 0]), feats, rd.RegimeConfig())

    def test_provided_transition_matrix_used(self):
        seq = np.array([0, 1, 0, 1])
        feats = self._dummy_features(len(seq))
        provided = np.array([[0.9, 0.1], [0.2, 0.8]])
        char = rd.characterize_regimes(feats, seq, feats, rd.RegimeConfig(),
                                       transition_matrix=provided)
        assert char["transition_matrix_source"] == "hmm"
        assert np.allclose(np.array(char["transition_matrix"]), provided)
