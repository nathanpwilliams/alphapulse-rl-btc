"""
trading_env.py - Gymnasium environment for the AlphaPulse RL BTC trading agent (v1)

Implements the MDP specified in docs/00_mdp_formulation.md (v1.1):
  - State: 24-dim vector (12 lagged 4H log-returns + 5 technical indicators
           + 3 portfolio features + 4 calendar features)
  - Action: continuous signed target position in [-1, 1]
  - Reward: log return of portfolio value with transaction costs and
            perpetual funding charges naturally included; components
            exposed in info dict for auditing
  - Episode: 360 4H bars (~60 days), terminated on 50% drawdown
  - Venue: Kraken Futures perpetual BTCUSD (testnet for paper trading)

Design principles:
  - DETERMINISTIC given (data, seed). Critical for reproducibility.
  - NO LOOK-AHEAD. All features at decision time t use only data <= t.
  - DECOMPOSABLE REWARD. info dict exposes price PnL, funding cost, and
    trade cost separately so we can audit reward composition in training.
  - PRECOMPUTED FEATURES. Indicators are computed once at construction,
    not per-step. step() is O(1) excluding the action.

Usage:
    from src.env.trading_env import TradingEnv, EnvConfig

    env = TradingEnv(data=df_4h_with_funding)
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(action=0.5)

Smoke test:
    python -m src.env.trading_env
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError(
        "gymnasium is required. Install with `pip install gymnasium`."
    ) from e


# ============================================================
# Configuration
# ============================================================

@dataclasses.dataclass
class EnvConfig:
    """Hyperparameters of the trading environment.

    Defaults track docs/00_mdp_formulation.md v1.1.
    """
    # Episode mechanics
    episode_length: int = 360          # 60 days * 6 4H-bars/day
    initial_balance: float = 1000.0    # Normalized
    drawdown_termination: float = 0.5  # Terminate if V < this fraction of V_0

    # Trading costs (Kraken Futures perp taker; conservative)
    transaction_cost: float = 0.0005   # 5 bps per side

    # Time conventions
    bar_hours: float = 4.0
    funding_period_hours: float = 1.0   # Kraken funding settles hourly
    # Feature parameters
    n_return_lags: int = 12            # 2 days of context at 4H
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    vol_period: int = 8                # ~32h of realized vol
    vol_z_window: int = 180            # ~30d for vol normalization

    # Optional features
    include_calendar: bool = True
    # Regime state feature (Level 3). When True, the observation is extended by the
    # per-bar causal regime posterior supplied to the env constructor. When False
    # (default), the env is byte-identical to the pre-regime version.
    include_regime: bool = False

    def __post_init__(self) -> None:
        assert self.episode_length > 0
        assert 0.0 <= self.transaction_cost < 0.1
        assert 0.0 < self.drawdown_termination < 1.0
        assert self.bar_hours > 0
        assert self.funding_period_hours > 0


# ============================================================
# Technical indicators (vectorized; no look-ahead)
# ============================================================

def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI. Returns values in [0, 100]."""
    delta = close.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _macd_histogram(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    """MACD histogram = MACD line - signal line. Same units as `close`."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder's Average True Range. Same units as `close`."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _bollinger_pctb(close: pd.Series, period: int, n_std: float) -> pd.Series:
    """Bollinger %B in approx [0, 1]: 0 = at lower band, 1 = at upper band."""
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    pctb = (close - lower) / (upper - lower).replace(0, np.nan)
    return pctb.fillna(0.5)


def _realized_vol(log_returns: pd.Series, period: int) -> pd.Series:
    """Rolling std of log returns over `period` bars."""
    return log_returns.rolling(period).std()


# ============================================================
# TradingEnv
# ============================================================

class TradingEnv(gym.Env):
    """Gymnasium env for single-asset BTC perpetual trading at 4H bars.

    See module docstring for full MDP specification and design rationale.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume", "funding_rate")

    def __init__(
        self,
        data: pd.DataFrame,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
        regime_features: np.ndarray | None = None,
    ):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode

        # Validate and stash data
        self._validate_data(data)
        self.data = data.copy()

        # Optional regime state feature (Level 3). The supplied array is the
        # per-bar regime posterior aligned POSITIONALLY to `data`. It is expected
        # to already be CAUSAL and one-bar-LAGGED (built upstream by
        # src/strategies/regime/features.build_causal_regime_posteriors): row t is
        # the posterior known as of bar t-1, using only data <= t-1. The env does
        # NOT compute or lag regimes itself - it just reads this array - so there is
        # no way for a look-ahead bug to enter here; causality is guaranteed by the
        # builder. Must be set before `_observation_dim()` is used below.
        if self.config.include_regime:
            if regime_features is None:
                raise ValueError(
                    "include_regime=True requires `regime_features` (the causal, "
                    "one-bar-lagged per-bar regime posterior array)."
                )
            regime_features = np.asarray(regime_features, dtype=np.float32)
            if regime_features.ndim != 2:
                raise ValueError(
                    f"regime_features must be 2D [n_bars, n_states]; got shape "
                    f"{regime_features.shape}."
                )
            if len(regime_features) != len(self.data):
                raise ValueError(
                    f"regime_features length {len(regime_features)} != data length "
                    f"{len(self.data)}; the posterior array must align to every bar."
                )
            self._regime_features = regime_features
            self._n_regime_features = int(regime_features.shape[1])
        else:
            self._regime_features = None
            self._n_regime_features = 0

        # Precompute every feature for the whole dataset (one-time cost)
        self._precompute_features()

        # Earliest index at which every feature is non-NaN
        self._min_start_idx = self._compute_min_valid_idx()
        # Latest valid start: must leave room for a full episode + 1 (we step into idx+1)
        self._max_start_idx = len(self.data) - self.config.episode_length - 1
        if self._max_start_idx <= self._min_start_idx:
            raise ValueError(
                f"Dataset too short: need at least "
                f"{self._min_start_idx + self.config.episode_length + 1} bars, "
                f"got {len(self.data)}."
            )

        # Gymnasium spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        n_obs = self._observation_dim()
        # Observation bounds intentionally permissive; features are scale-controlled
        # individually but we don't want PPO to crash on a single outlier bar.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )

        # Episode state (initialized properly in reset)
        self._current_idx: int = 0
        self._episode_start_idx: int = 0
        self._steps_taken: int = 0
        self._position: float = 0.0
        self._portfolio_value: float = self.config.initial_balance
        self._entry_price: float | None = None
        self._steps_in_position: int = 0
        self._history: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # Setup helpers
    # --------------------------------------------------------

    def _validate_data(self, df: pd.DataFrame) -> None:
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex (UTC recommended).")
        if not df.index.is_monotonic_increasing:
            raise ValueError("DataFrame index must be monotonically increasing.")
        if df["close"].isna().any() or (df["close"] <= 0).any():
            raise ValueError("`close` must be strictly positive and non-NaN.")
        if len(df) < 250:
            raise ValueError(
                f"Dataset too short for feature warmup: {len(df)} bars (need >=250)."
            )

    def _precompute_features(self) -> None:
        """Compute all features in-place on self.data. No look-ahead."""
        df = self.data
        cfg = self.config

        # Base series: log returns of close
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Lagged returns. At decision time t, lag-k = log_return at index t-(k-1).
        # That is, lag-1 = the return that just ended (observable), lag-2 = one before, etc.
        # All lags use only past or present information.
        for k in range(1, cfg.n_return_lags + 1):
            df[f"ret_lag_{k}"] = df["log_return"].shift(k - 1)

        # Technical indicators, each normalized to a roughly bounded range
        df["rsi"] = _rsi(df["close"], cfg.rsi_period) / 100.0
        df["macd_hist"] = _macd_histogram(
            df["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
        ) / df["close"]
        df["atr_norm"] = _atr(
            df["high"], df["low"], df["close"], cfg.atr_period
        ) / df["close"]
        df["bb_pctb"] = _bollinger_pctb(df["close"], cfg.bb_period, cfg.bb_std)

        # Realized vol normalized vs trailing 30d (log ratio, symmetric around 0)
        df["realized_vol"] = _realized_vol(df["log_return"], cfg.vol_period)
        med = df["realized_vol"].rolling(cfg.vol_z_window).median()
        df["realized_vol_log_ratio"] = np.log(
            (df["realized_vol"] + 1e-10) / (med + 1e-10)
        ).fillna(0.0)

        # Calendar features
        if cfg.include_calendar:
            bar_of_day = (df.index.hour // int(cfg.bar_hours)).astype(float)
            bars_per_day = 24.0 / cfg.bar_hours
            df["cal_sin_bod"] = np.sin(2 * np.pi * bar_of_day / bars_per_day)
            df["cal_cos_bod"] = np.cos(2 * np.pi * bar_of_day / bars_per_day)
            dow = df.index.dayofweek.astype(float)
            df["cal_sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
            df["cal_cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

        # Cache column groups for fast obs assembly
        self._feature_cols_lag = [f"ret_lag_{k}" for k in range(1, cfg.n_return_lags + 1)]
        self._feature_cols_tech = [
            "rsi", "macd_hist", "atr_norm", "bb_pctb", "realized_vol_log_ratio"
        ]
        self._feature_cols_cal = (
            ["cal_sin_bod", "cal_cos_bod", "cal_sin_dow", "cal_cos_dow"]
            if cfg.include_calendar else []
        )

    def _compute_min_valid_idx(self) -> int:
        """Earliest index at which every feature column is finite."""
        cfg = self.config
        warmup = max(
            cfg.n_return_lags,
            cfg.macd_slow + cfg.macd_signal,
            cfg.bb_period,
            cfg.rsi_period * 3,   # Wilder RMA needs warmup
            cfg.vol_z_window,
        )
        return warmup + 1

    def _observation_dim(self) -> int:
        cfg = self.config
        # base = 12 lags + 5 tech + 3 portfolio + 4 calendar; regime posterior (if
        # enabled) is appended last, adding one column per regime state.
        return (cfg.n_return_lags + 5 + 3
                + (4 if cfg.include_calendar else 0)
                + self._n_regime_features)

    # --------------------------------------------------------
    # Gymnasium API
    # --------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # Sample episode start uniformly in valid range
        self._episode_start_idx = int(self.np_random.integers(
            self._min_start_idx, self._max_start_idx + 1
        ))
        self._current_idx = self._episode_start_idx
        self._steps_taken = 0

        # Reset portfolio
        self._position = 0.0
        self._portfolio_value = self.config.initial_balance
        self._entry_price = None
        self._steps_in_position = 0
        self._history = []

        obs = self._get_obs()
        info = {
            "episode_start_idx": self._episode_start_idx,
            "episode_start_time": str(self.data.index[self._episode_start_idx]),
            "initial_balance": self._portfolio_value,
        }
        return obs, info

    def step(self, action):
        # Parse action into a scalar in [-1, 1]
        if hasattr(action, "__len__"):
            a = float(action[0])
        else:
            a = float(action)
        a = float(np.clip(a, -1.0, 1.0))
        prev_pos = self._position

        # Prices: trade executes at close[current_idx]; position is held
        # to close[current_idx + 1]; funding rate at current_idx + 1 applies.
        price_now = float(self.data["close"].iloc[self._current_idx])
        price_next = float(self.data["close"].iloc[self._current_idx + 1])
        funding_next = float(self.data["funding_rate"].iloc[self._current_idx + 1])

        # ---- Reward composition ----
        # 1. Trade cost charged on the position delta, at price_now.
        trade_size = abs(a - prev_pos)
        trade_cost_pct = self.config.transaction_cost * trade_size

        # 2. Price PnL on the held position over the bar.
        price_pct_change = (price_next - price_now) / price_now
        pos_pct_pnl = a * price_pct_change

        # 3. Funding cost. Convention: funding_rate > 0 means longs pay shorts.
        #    Funding is published per `funding_period_hours`; we hold for `bar_hours`.
        funding_pct = funding_next * a * (self.config.bar_hours / self.config.funding_period_hours)

        # Portfolio update: pay trade cost at price_now, then accrue P&L and funding.
        v_before = self._portfolio_value
        v_after_trade = v_before * (1.0 - trade_cost_pct)
        v_after_bar = v_after_trade * (1.0 + pos_pct_pnl - funding_pct)

        # Total reward is log return; safe-guard tiny/zero portfolio
        reward = float(np.log(max(v_after_bar / v_before, 1e-10)))

        # Update position tracking (entry_price, steps_in_position)
        EPS = 1e-9
        if abs(a) < EPS:
            self._entry_price = None
            self._steps_in_position = 0
        elif abs(prev_pos) < EPS or np.sign(prev_pos) != np.sign(a):
            # Opened from flat OR flipped sign
            self._entry_price = price_now
            self._steps_in_position = 1
        else:
            # Holding or adjusting size in same direction; keep original entry
            self._steps_in_position += 1

        # Build info dict with decomposed reward components for auditing
        info = {
            "reward": reward,
            "price_log_return": float(np.log(1.0 + pos_pct_pnl)) if (1.0 + pos_pct_pnl) > 0 else float("-inf"),
            "trade_cost_log": float(np.log(1.0 - trade_cost_pct)) if trade_cost_pct < 1.0 else float("-inf"),
            "funding_log": float(np.log(1.0 - funding_pct)) if funding_pct < 1.0 else float("-inf"),
            "action": a,
            "prev_position": prev_pos,
            "new_position": a,
            "trade_size": trade_size,
            "portfolio_value": v_after_bar,
            "price": price_next,
            "funding_rate": funding_next,
            "bar_idx": self._current_idx + 1,
        }
        self._history.append(info)

        # Commit state advance
        self._portfolio_value = v_after_bar
        self._position = a
        self._current_idx += 1
        self._steps_taken += 1

        # Termination criteria
        drawdown_terminated = (
            v_after_bar < self.config.initial_balance * self.config.drawdown_termination
        )
        episode_length_reached = self._steps_taken >= self.config.episode_length
        # gymnasium: `terminated` for absorbing states, `truncated` for time limits
        terminated = bool(drawdown_terminated)
        truncated = bool(episode_length_reached and not terminated)

        if terminated:
            info["termination_reason"] = "drawdown_50pct"
        elif truncated:
            info["termination_reason"] = "episode_length"

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------
    # Observation construction
    # --------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Assemble observation at self._current_idx. O(1)."""
        idx = self._current_idx
        row = self.data.iloc[idx]

        lag_returns = [float(row[c]) for c in self._feature_cols_lag]
        tech = [float(row[c]) for c in self._feature_cols_tech]
        cal = [float(row[c]) for c in self._feature_cols_cal]

        # Portfolio features
        pos = self._position
        if self._entry_price is not None and self._entry_price > 0:
            price_now = float(row["close"])
            pnl_pct = float(np.sign(pos) * (price_now - self._entry_price) / self._entry_price)
        else:
            pnl_pct = 0.0
        steps_log = float(np.log1p(self._steps_in_position))
        portfolio = [pos, pnl_pct, steps_log]

        # Regime posterior (Level 3) appended last, when enabled. Read straight from
        # the pre-lagged causal array at the current bar - no computation here.
        if self._n_regime_features > 0:
            regime = [float(v) for v in self._regime_features[idx]]
        else:
            regime = []

        obs = np.array(lag_returns + tech + portfolio + cal + regime, dtype=np.float32)

        # Safety net: any NaN/Inf becomes 0. Should not happen given
        # _min_start_idx guard, but defends against unexpected data quirks.
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    # --------------------------------------------------------
    # Rendering & introspection
    # --------------------------------------------------------

    def render(self):
        """Quick 3-panel plot of price, position, portfolio over the episode."""
        if not self._history:
            print("No history yet; call step() at least once before render().")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; render() is a no-op.")
            return

        df = pd.DataFrame(self._history)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        axes[0].plot(df["price"], label="Close")
        axes[0].set_ylabel("BTC price")
        axes[0].legend(loc="upper left")
        axes[0].set_title(
            f"Episode: idx {self._episode_start_idx} -> {self._current_idx}, "
            f"V_final={df['portfolio_value'].iloc[-1]:.2f}"
        )

        axes[1].plot(df["new_position"], color="orange", label="Position")
        axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
        axes[1].fill_between(
            range(len(df)), 0, df["new_position"], alpha=0.3, color="orange"
        )
        axes[1].set_ylabel("Position [-1, 1]")
        axes[1].set_ylim(-1.1, 1.1)

        axes[2].plot(df["portfolio_value"], color="green", label="Portfolio")
        axes[2].axhline(
            self.config.initial_balance, color="gray", linestyle="--", alpha=0.5,
            label="Initial",
        )
        axes[2].set_ylabel("Portfolio value")
        axes[2].set_xlabel("Step")
        axes[2].legend(loc="upper left")

        plt.tight_layout()
        plt.show()

    def get_episode_history(self) -> pd.DataFrame:
        """Return a DataFrame of all step() info dicts from the current episode."""
        return pd.DataFrame(self._history)

    def close(self):
        pass


# ============================================================
# Smoke test
# ============================================================

def _generate_synthetic_data(n_bars: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Generate plausible 4H BTC-like data for smoke testing."""
    rng = np.random.default_rng(seed)

    # GBM-like log returns with mild drift and 2% hourly vol scaled to 4H
    log_returns = rng.normal(0.0002, 0.022, n_bars)
    log_prices = np.cumsum(log_returns) + np.log(40000.0)
    close = np.exp(log_prices)

    # OHLC: synthesize intra-bar noise
    spread = close * 0.002
    open_ = close * (1.0 + rng.normal(0, 0.0008, n_bars))
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 1, n_bars)) * spread
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 1, n_bars)) * spread
    volume = rng.exponential(1000.0, n_bars)

    # Funding rate: small with regime persistence
    funding = np.cumsum(rng.normal(0, 1e-5, n_bars)) + rng.normal(0, 1e-4, n_bars)
    funding = np.clip(funding, -0.001, 0.001)

    idx = pd.date_range("2023-01-01", periods=n_bars, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "funding_rate": funding,
        },
        index=idx,
    )


def _smoke_test():
    print("=" * 60)
    print("TradingEnv smoke test")
    print("=" * 60)

    data = _generate_synthetic_data()
    print(f"Generated {len(data)} bars of synthetic data "
          f"({data.index[0]} -> {data.index[-1]})")

    env = TradingEnv(data=data)
    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Min start idx:     {env._min_start_idx}")
    print(f"Max start idx:     {env._max_start_idx}")
    print(f"Obs dim:           {env._observation_dim()}")

    # Reset and inspect
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape, "Obs shape mismatch"
    assert np.all(np.isfinite(obs)), "Initial obs contains non-finite values"
    print(f"\nInitial obs (first 8): {obs[:8]}")
    print(f"Initial info: {info}")

    # Run a few deterministic step types
    print("\n--- Test 1: Open long ---")
    obs, r, term, trunc, info = env.step(np.array([0.5]))
    print(f"  reward={r:+.6f}  pos={info['new_position']:+.2f}  "
          f"V={info['portfolio_value']:.4f}  trade_size={info['trade_size']:.2f}")

    print("--- Test 2: Hold ---")
    obs, r, term, trunc, info = env.step(np.array([0.5]))
    print(f"  reward={r:+.6f}  pos={info['new_position']:+.2f}  "
          f"V={info['portfolio_value']:.4f}  trade_size={info['trade_size']:.2f}")

    print("--- Test 3: Flip to short ---")
    obs, r, term, trunc, info = env.step(np.array([-0.7]))
    print(f"  reward={r:+.6f}  pos={info['new_position']:+.2f}  "
          f"V={info['portfolio_value']:.4f}  trade_size={info['trade_size']:.2f}")

    # Verify trade_size sanity check
    assert info["trade_size"] == 1.2, f"Expected trade size 1.2, got {info['trade_size']}"
    print("  [PASS] Position flip trade size = 1.2 (0.5 -> -0.7)")

    # Run a full random episode to verify no crashes / termination handling
    print("\n--- Test 4: Full random episode ---")
    obs, info = env.reset(seed=1)
    total_reward = 0.0
    step_count = 0
    while True:
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        total_reward += r
        step_count += 1
        if term or trunc:
            break
    print(f"  Episode ended after {step_count} steps "
          f"(reason: {info.get('termination_reason')})")
    print(f"  Cumulative log reward: {total_reward:+.4f}")
    print(f"  Final portfolio:       {info['portfolio_value']:.4f}")
    print(f"  Return (geometric):    {(np.exp(total_reward) - 1) * 100:+.2f}%")

    # Determinism check: same seed -> same trajectory
    print("\n--- Test 5: Determinism ---")
    env2 = TradingEnv(data=data)
    obs_a, _ = env.reset(seed=7)
    obs_b, _ = env2.reset(seed=7)
    assert np.allclose(obs_a, obs_b), "Reset with same seed should give identical obs"
    for _ in range(20):
        action = np.array([0.3])  # fixed action
        oa, ra, _, _, _ = env.step(action)
        ob, rb, _, _, _ = env2.step(action)
        assert np.allclose(oa, ob), "Deterministic step diverged"
        assert ra == rb, "Deterministic reward diverged"
    print("  [PASS] Same seed + same actions = identical trajectory")

    print("\n" + "=" * 60)
    print("Smoke test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _smoke_test()
