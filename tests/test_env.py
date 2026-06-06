"""
tests/test_env.py - Unit tests for the AlphaPulse RL trading environment.

Run with:
    pytest tests/test_env.py -v
    pytest tests/test_env.py -v --cov=src.env  (with coverage report)

These tests cover properties the smoke test in trading_env.py does NOT verify:
  - Data validation rejection paths
  - Observation shape and finiteness across configurations
  - Action handling and clipping
  - Reward component arithmetic (isolated by holding other variables constant)
  - Funding rate sign convention and magnitude
  - Episode termination paths (length truncation vs. drawdown)
  - Determinism across env instances with the same seed
  - No look-ahead in features

Design principle: each test isolates ONE property using minimal synthetic data
that holds other variables constant. When a test fails, the name and assertion
message should point directly at the broken property.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `src.env.trading_env` importable regardless of where pytest is invoked
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.env.trading_env import TradingEnv, EnvConfig  # noqa: E402


# ============================================================
# Synthetic data fixtures
# ============================================================

def make_random_data(n_bars: int = 1000, seed: int = 42) -> pd.DataFrame:
    """GBM-like prices, small random funding, plausible OHLC ranges."""
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
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "funding_rate": funding,
        },
        index=idx,
    )


def make_constant_price_data(
    n_bars: int = 1000,
    price: float = 40000.0,
    funding_rate: float = 0.0,
) -> pd.DataFrame:
    """Flat price, fixed funding - isolates non-price reward components."""
    idx = pd.date_range("2023-01-01", periods=n_bars, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.full(n_bars, price),
            "high": np.full(n_bars, price),
            "low": np.full(n_bars, price),
            "close": np.full(n_bars, price),
            "volume": np.full(n_bars, 1000.0),
            "funding_rate": np.full(n_bars, funding_rate),
        },
        index=idx,
    )


@pytest.fixture
def random_data():
    return make_random_data()


@pytest.fixture
def env(random_data):
    return TradingEnv(data=random_data)


# ============================================================
# Data validation - the env should reject bad inputs LOUDLY
# ============================================================

class TestDataValidation:
    """The env must reject malformed input with a clear ValueError, not crash
    later in a confusing way during reset() or step()."""

    def test_rejects_missing_required_column(self, random_data):
        df = random_data.drop(columns=["funding_rate"])
        with pytest.raises(ValueError, match="[Mm]issing required columns"):
            TradingEnv(data=df)

    def test_rejects_non_datetime_index(self, random_data):
        df = random_data.reset_index(drop=True)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            TradingEnv(data=df)

    def test_rejects_non_monotonic_index(self, random_data):
        df = random_data.iloc[::-1].copy()
        with pytest.raises(ValueError, match="monotonically"):
            TradingEnv(data=df)

    def test_rejects_negative_close_prices(self, random_data):
        df = random_data.copy()
        df.iloc[5, df.columns.get_loc("close")] = -100.0
        with pytest.raises(ValueError, match="positive"):
            TradingEnv(data=df)

    def test_rejects_too_short_dataset(self):
        df = make_random_data(n_bars=100)
        with pytest.raises(ValueError, match="too short"):
            TradingEnv(data=df)


# ============================================================
# Observation space - shape and finiteness guarantees
# ============================================================

class TestObservation:

    def test_obs_dim_with_calendar(self, env):
        # 12 lag returns + 5 tech + 3 portfolio + 4 calendar = 24
        assert env.observation_space.shape == (24,)

    def test_obs_dim_without_calendar(self):
        cfg = EnvConfig(include_calendar=False)
        env = TradingEnv(data=make_random_data(), config=cfg)
        # 12 + 5 + 3 = 20
        assert env.observation_space.shape == (20,)

    def test_obs_finite_at_reset(self, env):
        obs, _ = env.reset(seed=0)
        assert np.all(np.isfinite(obs)), "Initial obs contains NaN or Inf"

    def test_obs_finite_throughout_random_episode(self, env):
        env.reset(seed=0)
        for _ in range(100):
            action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            assert np.all(np.isfinite(obs)), "Mid-episode obs contains NaN or Inf"
            if term or trunc:
                break

    def test_initial_portfolio_features_are_zero(self, env):
        """At episode start: position=0, unrealized PnL=0, steps_in_position=0."""
        obs, _ = env.reset(seed=0)
        # Portfolio features sit after 12 lags + 5 tech = indices 17, 18, 19
        pos, pnl, steps = obs[17], obs[18], obs[19]
        assert pos == 0.0
        assert pnl == 0.0
        assert steps == 0.0


# ============================================================
# Action handling - clipping, no-op trades, position tracking
# ============================================================

class TestAction:

    def test_action_below_range_clips_to_minus_one(self, env):
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([-5.0]))
        assert info["new_position"] == -1.0

    def test_action_above_range_clips_to_plus_one(self, env):
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([3.0]))
        assert info["new_position"] == 1.0

    def test_zero_action_produces_zero_trade_size(self, env):
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.0]))
        assert info["trade_size"] == 0.0
        assert info["new_position"] == 0.0

    def test_holding_same_position_produces_zero_trade_size(self, env):
        env.reset(seed=0)
        env.step(np.array([0.5]))                       # open
        _, _, _, _, info = env.step(np.array([0.5]))    # hold
        assert info["trade_size"] == 0.0

    def test_scalar_action_accepted(self, env):
        """API tolerance: PPO sometimes passes scalars, sometimes arrays."""
        env.reset(seed=0)
        _, _, _, _, info = env.step(0.5)
        assert info["new_position"] == 0.5


# ============================================================
# Reward components - isolated using constant-price data
# ============================================================

class TestRewardComponents:
    """Use constant-price + zero-funding data to isolate the cost component,
    constant-price + nonzero-funding to isolate the funding component."""

    def test_trade_cost_proportional_to_position_delta(self):
        """Going 0 -> 0.5 with 10bps cost: reward = log(1 - 0.001 * 0.5)."""
        data = make_constant_price_data(funding_rate=0.0)
        cfg = EnvConfig(transaction_cost=0.001)
        env = TradingEnv(data=data, config=cfg)
        env.reset(seed=0)

        _, reward, _, _, info = env.step(np.array([0.5]))
        assert info["trade_size"] == 0.5

        expected = np.log(1.0 - 0.001 * 0.5)
        assert reward == pytest.approx(expected, rel=1e-6)

    def test_position_flip_costs_full_notional(self):
        """Flipping +0.5 -> -0.5 must charge cost on 1.0 notional (the sum)."""
        data = make_constant_price_data(funding_rate=0.0)
        cfg = EnvConfig(transaction_cost=0.001)
        env = TradingEnv(data=data, config=cfg)
        env.reset(seed=0)

        env.step(np.array([0.5]))
        _, _, _, _, info = env.step(np.array([-0.5]))
        assert info["trade_size"] == pytest.approx(1.0, abs=1e-9)

    def test_pure_hold_with_constant_price_zero_funding_is_zero_reward(self):
        """No trade + no price change + no funding -> reward must be exactly 0."""
        data = make_constant_price_data(funding_rate=0.0)
        env = TradingEnv(data=data)
        env.reset(seed=0)

        env.step(np.array([0.5]))                          # open (incurs cost)
        _, reward, _, _, info = env.step(np.array([0.5]))  # pure hold
        assert info["trade_size"] == 0.0
        assert reward == pytest.approx(0.0, abs=1e-9)


# ============================================================
# Funding mechanics - sign convention and magnitude
# ============================================================

class TestFunding:
    """Sign convention: funding_rate > 0 means longs pay shorts.
       Magnitude: funding_pct = funding_rate * position * (bar_h / funding_h).

    These tests use transaction_cost=0 to isolate the funding effect."""

    def test_long_position_pays_positive_funding(self):
        data = make_constant_price_data(funding_rate=0.001)
        cfg = EnvConfig(transaction_cost=0.0)
        env = TradingEnv(data=data, config=cfg)
        env.reset(seed=0)

        env.step(np.array([1.0]))                          # open full long
        _, reward, _, _, _ = env.step(np.array([1.0]))     # hold
        assert reward < 0, "Long + positive funding must reduce reward"

    def test_short_position_receives_positive_funding(self):
        data = make_constant_price_data(funding_rate=0.001)
        cfg = EnvConfig(transaction_cost=0.0)
        env = TradingEnv(data=data, config=cfg)
        env.reset(seed=0)

        env.step(np.array([-1.0]))                         # open full short
        _, reward, _, _, _ = env.step(np.array([-1.0]))    # hold
        assert reward > 0, "Short + positive funding must increase reward"

    def test_flat_position_has_zero_funding_exposure(self):
        data = make_constant_price_data(funding_rate=0.001)
        cfg = EnvConfig(transaction_cost=0.0)
        env = TradingEnv(data=data, config=cfg)
        env.reset(seed=0)

        _, reward, _, _, _ = env.step(np.array([0.0]))
        assert reward == pytest.approx(0.0, abs=1e-9)

    def test_funding_magnitude_matches_bar_fraction(self):
        """For 4h bars with 8h funding period: funding_pct should be HALF
        the published rate when fully long. Pins down the exact arithmetic."""
        funding_rate = 0.001  # 10 bps per 8h
        data = make_constant_price_data(funding_rate=funding_rate)
        cfg = EnvConfig(
            transaction_cost=0.0, bar_hours=4.0, funding_period_hours=8.0,
        )
        env = TradingEnv(data=data, config=cfg)
        env.reset(seed=0)

        env.step(np.array([1.0]))                              # open
        _, reward, _, _, _ = env.step(np.array([1.0]))         # hold full long

        expected_funding_pct = funding_rate * 1.0 * (4.0 / 8.0)
        expected_reward = np.log(1.0 - expected_funding_pct)
        assert reward == pytest.approx(expected_reward, rel=1e-6)


# ============================================================
# Episode termination - truncation vs. drawdown termination
# ============================================================

class TestEpisodeTermination:

    def test_episode_truncates_at_configured_length(self):
        """With a flat policy and no drawdown trigger, episode should end
        via truncation exactly at the configured length."""
        data = make_random_data(n_bars=1000)
        cfg = EnvConfig(episode_length=50)
        env = TradingEnv(data=data, config=cfg)
        env.reset(seed=0)

        steps = 0
        term, trunc = False, False
        while not (term or trunc):
            _, _, term, trunc, _ = env.step(np.array([0.0]))
            steps += 1

        assert steps == 50
        assert trunc is True
        assert term is False, "Flat policy should not trigger drawdown termination"

    def test_drawdown_termination_fires_below_threshold(self):
        """Construct a deterministic 70% crash and verify drawdown termination."""
        n_bars = 1000
        crash_idx = 700

        data = make_constant_price_data(n_bars=n_bars, price=40000.0)
        # Drop price 70% at crash_idx; that breaches the 50% V threshold for full long
        for col in ("open", "high", "low", "close"):
            data.iloc[crash_idx:, data.columns.get_loc(col)] = 12000.0

        cfg = EnvConfig(
            episode_length=200, drawdown_termination=0.5, transaction_cost=0.0,
        )
        env = TradingEnv(data=data, config=cfg)

        # Reset, then deterministically place the env just before the crash.
        # (Bypasses random episode-start sampling for test determinism.)
        env.reset(seed=0)
        env._current_idx = crash_idx - 2
        env._episode_start_idx = crash_idx - 2
        env._steps_taken = 0
        env._position = 0.0
        env._portfolio_value = cfg.initial_balance

        env.step(np.array([1.0]))  # open full long (zero cost since c=0)

        # Next step traverses the crash bar -> V drops to 30% of initial
        _, _, term, _, info = env.step(np.array([1.0]))

        assert term is True
        assert info.get("termination_reason") == "drawdown_50pct"


# ============================================================
# Determinism - critical for ML reproducibility
# ============================================================

class TestDeterminism:

    def test_same_seed_produces_identical_trajectory(self, random_data):
        env1 = TradingEnv(data=random_data)
        env2 = TradingEnv(data=random_data)

        obs1, _ = env1.reset(seed=7)
        obs2, _ = env2.reset(seed=7)
        np.testing.assert_allclose(obs1, obs2)

        for a in [0.3, -0.5, 0.8, 0.0, -0.2]:
            o1, r1, _, _, _ = env1.step(np.array([a]))
            o2, r2, _, _, _ = env2.step(np.array([a]))
            np.testing.assert_allclose(o1, o2)
            assert r1 == r2

    def test_different_seeds_produce_different_episode_starts(self, random_data):
        env1 = TradingEnv(data=random_data)
        env2 = TradingEnv(data=random_data)
        env1.reset(seed=1)
        env2.reset(seed=2)
        # With ~300 valid start indices, P(collision) is ~1/300 - safe to assert
        assert env1._episode_start_idx != env2._episode_start_idx


# ============================================================
# No look-ahead - the single most important correctness property
# ============================================================

class TestLookAhead:
    """If observations at time T depend on data after T, we have leakage.
    A leaked future would let the agent learn an unrealizable policy
    that catastrophically fails in live trading."""

    def test_obs_unchanged_when_future_modified(self, random_data):
        """Construct two datasets identical up to time T, divergent after.
        Observation at time T must be byte-identical between them."""
        T = 300

        data_a = random_data.copy()
        data_b = random_data.copy()
        # Mangle EVERY column after T in data_b
        for col in ("open", "high", "low", "close"):
            data_b.iloc[T + 1:, data_b.columns.get_loc(col)] *= 2.0
        data_b.iloc[T + 1:, data_b.columns.get_loc("funding_rate")] += 0.01
        data_b.iloc[T + 1:, data_b.columns.get_loc("volume")] *= 10.0

        env_a = TradingEnv(data=data_a)
        env_b = TradingEnv(data=data_b)
        env_a.reset(seed=0)
        env_b.reset(seed=0)
        env_a._current_idx = T
        env_b._current_idx = T

        obs_a = env_a._get_obs()
        obs_b = env_b._get_obs()

        np.testing.assert_allclose(
            obs_a, obs_b,
            err_msg=(
                "Observation at time T was affected by data AFTER T. "
                "This is a look-ahead bug - the agent can see the future."
            ),
        )
