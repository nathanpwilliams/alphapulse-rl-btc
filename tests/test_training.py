"""
tests/test_training.py - Smoke tests for the PPO training scaffold.

These tests validate the training PIPELINE: data loading -> env construction
-> PPO setup -> training loop -> save/load. They do NOT test whether PPO
converges to a profitable policy. That's an experiment, not a unit test.

All tests use synthetic data with tiny training budgets (a few hundred steps)
so the entire suite runs in seconds, not minutes.

Run:
    pytest tests/test_training.py -v
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from stable_baselines3 import PPO

from src.training import train as train_mod
from src.training.train import (
    build_ppo_model,
    load_and_split_data,
    make_env,
    make_vec_env,
    save_run_config,
)
from src.env.trading_env import EnvConfig, TradingEnv


# ============================================================
# Synthetic data helper
# ============================================================

def make_synthetic_dataset(n_bars: int = 1600, seed: int = 42) -> pd.DataFrame:
    """Build a plausible env-compatible dataset for offline tests.

    Equivalent in shape to what build_dataset() produces from Kraken, but
    generated locally so tests don't depend on network or the API.
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0001, 0.02, n_bars)
    close = np.exp(np.cumsum(log_returns) + np.log(40000.0))
    spread = close * 0.002
    open_ = close * (1.0 + rng.normal(0, 0.0008, n_bars))
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 1, n_bars)) * spread
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 1, n_bars)) * spread
    volume = rng.exponential(1000.0, n_bars)
    funding = rng.normal(0, 1e-6, n_bars).clip(-1e-4, 1e-4)
    idx = pd.date_range("2025-01-01", periods=n_bars, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "funding_rate": funding,
        },
        index=idx,
    )


# ============================================================
# Train/val data split
# ============================================================

class TestDataSplit:
    """Verify train/val is chronological with no overlap and correct sizing."""

    def test_split_is_chronological_no_overlap(self):
        synthetic = make_synthetic_dataset(n_bars=1600)
        with patch.object(train_mod, "build_dataset", return_value=synthetic):
            train_df, val_df = load_and_split_data(
                "2025-01-01", "2025-08-01", val_fraction=0.25,
            )
        assert train_df.index[-1] < val_df.index[0], \
            "Train data leaks into val time period (look-ahead bug)"

    def test_split_proportions_correct(self):
        synthetic = make_synthetic_dataset(n_bars=1000)
        with patch.object(train_mod, "build_dataset", return_value=synthetic):
            train_df, val_df = load_and_split_data(
                "2025-01-01", "2025-08-01", val_fraction=0.25,
            )
        assert len(train_df) == 750
        assert len(val_df) == 250

    def test_no_bars_lost_in_split(self):
        synthetic = make_synthetic_dataset(n_bars=1200)
        with patch.object(train_mod, "build_dataset", return_value=synthetic):
            train_df, val_df = load_and_split_data(
                "2025-01-01", "2025-08-01", val_fraction=0.2,
            )
        assert len(train_df) + len(val_df) == 1200


# ============================================================
# Env construction
# ============================================================

class TestEnvConstruction:

    def test_make_env_returns_callable_factory(self):
        """make_env should return a callable, not an env directly."""
        data = make_synthetic_dataset()
        env_fn = make_env(data, seed=0, rank=0)
        assert callable(env_fn), "make_env must return a callable factory"

    def test_factory_produces_valid_env(self):
        data = make_synthetic_dataset()
        env_fn = make_env(data, seed=0, rank=0)
        env = env_fn()
        assert hasattr(env, "step")
        assert hasattr(env, "reset")
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")

    def test_vec_env_constructs_n_envs(self):
        data = make_synthetic_dataset()
        vec_env = make_vec_env(data, n_envs=4, seed=0)
        assert vec_env.num_envs == 4

    def test_parallel_envs_have_different_episode_starts(self):
        """seed + rank pattern must produce distinct episode start indices."""
        data = make_synthetic_dataset()
        vec_env = make_vec_env(data, n_envs=4, seed=42)
        start_indices = [env._episode_start_idx for env in vec_env.envs]
        # With ~650 valid start positions, 4 envs colliding is extremely unlikely
        assert len(set(start_indices)) >= 2, \
            f"Parallel envs got identical episode starts: {start_indices}"

    def test_custom_env_config_passes_through(self):
        """Verify EnvConfig param is honored (used for val env with shorter episodes)."""
        data = make_synthetic_dataset()
        config = EnvConfig(episode_length=120)
        vec_env = make_vec_env(data, n_envs=1, seed=0, config=config)
        # Unwrap to underlying TradingEnv to check config
        env = vec_env.envs[0]
        assert env.config.episode_length == 120


# ============================================================
# PPO model construction
# ============================================================

class TestPPOConstruction:

    def test_model_builds_with_expected_hyperparams(self, tmp_path):
        data = make_synthetic_dataset()
        env = make_vec_env(data, n_envs=2, seed=0)
        model = build_ppo_model(env, log_dir=tmp_path, seed=0)
        assert isinstance(model, PPO)
        assert model.n_steps == 2048
        assert model.gamma == 0.95
        assert model.ent_coef == 0.01
        assert model.gae_lambda == 0.95

    def test_policy_architecture_has_64_unit_layers(self, tmp_path):
        """Verify the 2x64 MLP architecture per design doc."""
        data = make_synthetic_dataset()
        env = make_vec_env(data, n_envs=2, seed=0)
        model = build_ppo_model(env, log_dir=tmp_path, seed=0)
        # Policy network str representation should mention the hidden layer sizes
        policy_repr = str(model.policy)
        assert "64" in policy_repr, "Expected 64-unit hidden layers in policy"


# ============================================================
# Smoke training run (tiny budget; verifies the loop doesn't crash)
# ============================================================

class TestTrainingSmoke:

    def test_short_training_completes_without_error(self, tmp_path):
        data = make_synthetic_dataset()
        env = make_vec_env(data, n_envs=2, seed=0)
        model = build_ppo_model(env, log_dir=tmp_path, seed=0)
        # Tiny budget; SB3 PPO needs at least n_envs * n_steps to fill the rollout
        # buffer once, but learn() handles partial buffers gracefully if our
        # total < n_envs * n_steps. We just verify it runs.
        model.learn(total_timesteps=500)
        assert model.num_timesteps >= 500

    def test_model_save_and_load_roundtrip(self, tmp_path):
        data = make_synthetic_dataset()
        env = make_vec_env(data, n_envs=2, seed=0)
        model = build_ppo_model(env, log_dir=tmp_path, seed=0)
        model.learn(total_timesteps=500)

        save_path = tmp_path / "test_model.zip"
        model.save(str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

        loaded = PPO.load(str(save_path))
        # Loaded model must be able to predict actions on real observations
        test_env = TradingEnv(data=data)
        obs, _ = test_env.reset(seed=0)
        action, _ = loaded.predict(obs, deterministic=True)
        assert action.shape == (1,)
        assert -1.0 <= float(action[0]) <= 1.0


# ============================================================
# Run config persistence
# ============================================================

class TestRunConfig:

    def test_save_run_config_writes_valid_json(self, tmp_path):
        args = argparse.Namespace(
            timesteps=1000, n_envs=2, seed=42,
            data_start="2025-01-01", data_end="2025-06-01",
            val_fraction=0.25, val_episode_length=120,
            checkpoint_freq=500, eval_freq=200,
            run_id="test_run", net_arch="64,64",
        )
        save_run_config(tmp_path, args, train_n=750, val_n=250)

        config_path = tmp_path / "run_config.json"
        assert config_path.exists()

        config = json.loads(config_path.read_text())
        assert config["timesteps"] == 1000
        assert config["seed"] == 42
        assert config["train_bars"] == 750
        assert config["val_bars"] == 250
        assert config["run_id"] == "test_run"
        assert "timestamp" in config
        assert config["net_arch"] == "64,64"
