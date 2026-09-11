"""
tests/test_regime_features.py - Unit tests for Level 3 regime state features.

Covers src/strategies/regime/features.py (causal, lagged posteriors), the soft
walk-forward path in backtest.py, the env's optional regime feature, and the
training wiring. Synthetic data, no network.

The non-negotiable guards: the regime posterior fed into bar t's observation uses
only data <= t-1 (walk-forward + one-bar lag). If those tests are hard to satisfy,
the causality is not clean.

Run:
    pytest tests/test_regime_features.py -v
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.env.trading_env import EnvConfig, TradingEnv  # noqa: E402
from src.strategies.regime import backtest as bt  # noqa: E402
from src.strategies.regime.detector import RegimeConfig, compute_features  # noqa: E402
from src.strategies.regime.features import build_causal_regime_posteriors  # noqa: E402
from src.training import train as train_mod  # noqa: E402


# ============================================================
# Synthetic builders
# ============================================================

def synth_features(n: int, seed: int = 0, block: int = 30) -> pd.DataFrame:
    """Two clearly-separated regimes so a 2-state HMM fits stably."""
    rng = np.random.default_rng(seed)
    lr, vol, tr = np.empty(n), np.empty(n), np.empty(n)
    for b in range(0, n, block):
        sl = slice(b, min(b + block, n))
        size = min(b + block, n) - b
        if (b // block) % 2 == 1:
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
    rng = np.random.default_rng(seed)
    rets = np.empty(n)
    for b in range(0, n, block):
        sl = slice(b, min(b + block, n))
        size = min(b + block, n) - b
        rets[sl] = (rng.normal(0.0, 0.03, size) if (b // block) % 2 == 1
                    else rng.normal(0.0015, 0.005, size))
    close = 40000.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"close": close}, index=idx)


def make_env_data(n_bars: int = 400, seed: int = 42) -> pd.DataFrame:
    """Full OHLCV+funding frame the env accepts."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0001, 0.02, n_bars)
    close = np.exp(np.cumsum(log_returns) + np.log(40000.0))
    spread = close * 0.002
    open_ = close * (1.0 + rng.normal(0, 0.0008, n_bars))
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 1, n_bars)) * spread
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 1, n_bars)) * spread
    volume = rng.exponential(1000.0, n_bars)
    funding = rng.normal(0, 1e-4, n_bars).clip(-0.001, 0.001)
    idx = pd.date_range("2023-01-01", periods=n_bars, freq="4h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": volume, "funding_rate": funding}, index=idx)


def fast_config(**kw) -> bt.RegimeBacktestConfig:
    base = dict(n_states=2, refit_every=40, min_train_bars=120, n_restarts=2, n_iter=50)
    base.update(kw)
    return bt.RegimeBacktestConfig(**base)


# ============================================================
# LOOK-AHEAD GUARD (non-negotiable) — soft posteriors
# ============================================================

class TestPosteriorNoLookAhead:

    @pytest.mark.parametrize("offset", [2, 45])
    def test_posterior_depends_only_on_past(self, offset):
        feats = synth_features(240, seed=1)
        cfg = fast_config()
        base = bt.assign_regime_posteriors_walk_forward(feats, cfg)["posteriors"]

        t_pos = 140
        t_ts = feats.index[t_pos]
        perturbed = feats.copy()
        perturbed.iloc[t_pos + offset] = perturbed.iloc[t_pos + offset] * 10.0 + 5.0
        after = bt.assign_regime_posteriors_walk_forward(perturbed, cfg)["posteriors"]

        # Posteriors up to and including bar t must be byte-identical.
        pd.testing.assert_frame_equal(base.loc[:t_ts], after.loc[:t_ts])

    def test_determinism_same_seed(self):
        feats = synth_features(200, seed=2)
        cfg = fast_config()
        a = bt.assign_regime_posteriors_walk_forward(feats, cfg)["posteriors"]
        b = bt.assign_regime_posteriors_walk_forward(feats, cfg)["posteriors"]
        pd.testing.assert_frame_equal(a, b)


# ============================================================
# Posterior validity + soft walk-forward shape
# ============================================================

class TestPosteriorValidity:

    def test_rows_are_distributions(self):
        feats = synth_features(220, seed=3)
        cfg = fast_config()
        post = bt.assign_regime_posteriors_walk_forward(feats, cfg)["posteriors"]
        arr = post.to_numpy()
        assert (arr >= 0).all()
        assert np.allclose(arr.sum(axis=1), 1.0)
        assert arr.shape[1] == 2  # n_states

    def test_labels_start_at_min_train_bars(self):
        feats = synth_features(220, seed=4)
        cfg = fast_config(min_train_bars=120, refit_every=40)
        post = bt.assign_regime_posteriors_walk_forward(feats, cfg)["posteriors"]
        assert post.index[0] == feats.index[120]


# ============================================================
# Causal builder: alignment, lag, warmup
# ============================================================

class TestBuildCausalPosteriors:

    def test_shape_and_alignment_to_data(self):
        ohlcv = synth_ohlcv(320, seed=5)
        cfg = fast_config(min_train_bars=120, refit_every=40)
        full, n_states = build_causal_regime_posteriors(ohlcv, cfg)
        assert full.shape == (len(ohlcv), n_states)
        assert n_states == 2

    def test_all_rows_are_valid_distributions(self):
        ohlcv = synth_ohlcv(320, seed=6)
        cfg = fast_config(min_train_bars=120, refit_every=40)
        full, _ = build_causal_regime_posteriors(ohlcv, cfg)
        assert (full >= 0).all()
        assert np.allclose(full.sum(axis=1), 1.0)  # every bar, incl. warmup

    def test_warmup_and_feature_dropped_bars_are_uniform(self):
        ohlcv = synth_ohlcv(320, seed=7)
        cfg = fast_config(min_train_bars=120, refit_every=40)
        full, n_states = build_causal_regime_posteriors(ohlcv, cfg)
        uniform = np.full(n_states, 1.0 / n_states)
        # Bar 0 is a feature-warmup bar (dropped by compute_features) -> uniform.
        assert np.allclose(full[0], uniform)

    def test_one_bar_lag(self):
        """full at data-position of feature bar j equals the causal posterior at
        feature bar j-1 (one-bar detection lag)."""
        ohlcv = synth_ohlcv(320, seed=8)
        cfg = fast_config(min_train_bars=120, refit_every=40)
        rc = RegimeConfig(vol_window=cfg.vol_window, trend_window=cfg.trend_window)
        feats = compute_features(ohlcv, rc)
        raw = bt.assign_regime_posteriors_walk_forward(feats, cfg)["posteriors"]
        full, n_states = build_causal_regime_posteriors(ohlcv, cfg)

        j = 130  # a labeled feature bar whose predecessor is also labeled
        data_pos_j = ohlcv.index.get_loc(feats.index[j])
        expected = raw.loc[feats.index[j - 1]].to_numpy()
        assert np.allclose(full[data_pos_j], expected)

        # The FIRST labeled feature bar (min_train_bars) has no labeled predecessor
        # -> its lagged feature is uniform.
        data_pos_first = ohlcv.index.get_loc(feats.index[cfg.min_train_bars])
        assert np.allclose(full[data_pos_first], np.full(n_states, 1.0 / n_states))


# ============================================================
# Env: baseline unchanged + augmented
# ============================================================

class TestEnvRegimeFeature:

    def test_baseline_dim_unchanged(self):
        env = TradingEnv(data=make_env_data(700), config=EnvConfig())
        assert env.observation_space.shape == (24,)
        assert env._n_regime_features == 0

    def test_augmented_dim_is_base_plus_n_states(self):
        data = make_env_data(700)
        n_states = 3
        arr = np.full((len(data), n_states), 1.0 / n_states, dtype=np.float32)
        env = TradingEnv(data=data, config=EnvConfig(include_regime=True),
                         regime_features=arr)
        assert env.observation_space.shape == (24 + n_states,)

    def test_regime_columns_match_supplied_array(self):
        data = make_env_data(700)
        rng = np.random.default_rng(0)
        raw = rng.random((len(data), 3))
        arr = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)  # distributions
        env = TradingEnv(data=data, config=EnvConfig(include_regime=True),
                         regime_features=arr)
        env.reset(seed=0)
        env._current_idx = 250
        obs = env._get_obs()
        # Regime posterior is the LAST n_states entries of the observation.
        assert np.allclose(obs[-3:], arr[250], atol=1e-6)

    def test_missing_regime_features_raises(self):
        with pytest.raises(ValueError, match="requires `regime_features`"):
            TradingEnv(data=make_env_data(), config=EnvConfig(include_regime=True))

    def test_length_mismatch_raises(self):
        data = make_env_data(400)
        arr = np.full((len(data) - 5, 3), 1 / 3, dtype=np.float32)
        with pytest.raises(ValueError, match="align to every bar"):
            TradingEnv(data=data, config=EnvConfig(include_regime=True),
                       regime_features=arr)

    def test_augmented_obs_finite_through_episode(self):
        data = make_env_data(500)
        arr = np.full((len(data), 2), 0.5, dtype=np.float32)
        env = TradingEnv(data=data, config=EnvConfig(include_regime=True, episode_length=50),
                         regime_features=arr)
        env.reset(seed=0)
        for _ in range(50):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            assert np.all(np.isfinite(obs))
            if term or trunc:
                break


# ============================================================
# Training wiring
# ============================================================

class TestTrainingWiring:

    def _args(self, tmp_path, run_id, include_regime):
        return argparse.Namespace(
            timesteps=200, n_envs=1, seed=42,
            data_start="2023-01-01", data_end="2023-09-01",
            val_fraction=0.25, val_episode_length=120,
            checkpoint_freq=10_000, eval_freq=10_000,
            net_arch="32,32", run_id=run_id,
            include_regime=include_regime, regime_n_states=2,
        )

    def test_baseline_run_records_flag_false(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data = make_env_data(1600)
        with patch.object(train_mod, "build_dataset", return_value=data):
            train_mod.train(self._args(tmp_path, "L3_base_test", include_regime=False))
        cfg = json.loads((tmp_path / "models" / "L3_base_test" / "run_config.json").read_text())
        assert cfg["include_regime"] is False
        assert cfg["regime_n_states"] is None

    def test_regime_run_records_flag_and_builds_extended_env(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data = make_env_data(1600)
        posteriors = np.full((len(data), 2), 0.5, dtype=np.float32)

        with patch.object(train_mod, "build_dataset", return_value=data), \
             patch.object(train_mod, "build_causal_regime_posteriors",
                          return_value=(posteriors, 2)) as mock_build:
            train_mod.train(self._args(tmp_path, "L3_regime_test", include_regime=True))

        mock_build.assert_called_once()  # regime path was taken
        cfg = json.loads((tmp_path / "models" / "L3_regime_test" / "run_config.json").read_text())
        assert cfg["include_regime"] is True
        assert cfg["regime_n_states"] == 2
