"""
src/training/train.py - PPO training for the AlphaPulse RL trading agent.

Usage:
    # Quick smoke test (~5-10 min on CPU)
    python -m src.training.train --timesteps 50000

    # Full training run (overnight)
    python -m src.training.train --timesteps 2000000 --n-envs 8

    # While training, in another terminal:
    tensorboard --logdir runs/

Design principles:
    DETERMINISTIC. Same seed + same data = same training trajectory.
    SEPARATE TRAIN/VAL. Chronological split. Val is strictly later in time
        than train, so the agent never sees val data during training. This
        is the only honest way to split time series.
    CHECKPOINTED. Periodic saves so a Ctrl+C or crashed run isn't wasted.
    LOGGED. Every loss, metric, episode reward goes to TensorBoard.
    REPRODUCIBLE. Run config saved as JSON alongside the model.

Output structure:
    runs/{run_id}/         TensorBoard logs + monitor CSVs
    models/{run_id}/       Periodic checkpoints + best model + final model
    models/{run_id}/run_config.json   Exact args used for this run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Project imports
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.load_kraken import build_dataset
from src.env.trading_env import TradingEnv, EnvConfig


# ============================================================
# Defaults
# ============================================================

# Data window: ~9 months ending ~6 weeks before today.
# Tuned to stay safely inside Kraken's ~1-year funding history window.
DEFAULT_DATA_START = "2025-08-01"
DEFAULT_DATA_END = "2026-05-01"

DEFAULT_VAL_FRACTION = 0.25
DEFAULT_TIMESTEPS = 50_000
DEFAULT_N_ENVS = 4
DEFAULT_SEED = 42

# Val episodes are shorter than train episodes so the smaller val data slice
# can fit multiple evaluation episodes. Different episode_length, same env.
DEFAULT_VAL_EPISODE_LENGTH = 120  # 20 days at 4H bars

logger = logging.getLogger(__name__)


# ============================================================
# Data preparation
# ============================================================

def load_and_split_data(
    start: str,
    end: str,
    val_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load BTC perpetual data from Kraken and split chronologically.

    The split is strictly time-ordered: train is earlier, val is later.
    No random shuffling - that would inject a look-ahead bug for time series.

    Returns:
        (train_df, val_df) where train_df.index[-1] < val_df.index[0]
    """
    df = build_dataset(start=start, end=end)
    n = len(df)
    split_idx = int(n * (1.0 - val_fraction))
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    logger.info(
        "Data split: train=%d bars (%s -> %s), val=%d bars (%s -> %s)",
        len(train_df), train_df.index[0], train_df.index[-1],
        len(val_df), val_df.index[0], val_df.index[-1],
    )
    return train_df, val_df


# ============================================================
# Environment construction
# ============================================================

def make_env(
    data: pd.DataFrame,
    seed: int,
    rank: int,
    monitor_dir: Optional[Path] = None,
    config: Optional[EnvConfig] = None,
):
    """Factory for a single env instance.

    SB3's vector envs expect a callable that returns an env, not an env
    directly. This lets the wrapper defer construction until it decides
    how many parallel instances to spin up.

    Each parallel env gets a different RNG seed (seed + rank) so their
    episode starts diverge; otherwise n parallel rollouts would all sample
    the same episode and learning would be redundant.
    """
    def _init():
        env = TradingEnv(data=data, config=config)
        env.reset(seed=seed + rank)
        if monitor_dir is not None:
            monitor_dir.mkdir(parents=True, exist_ok=True)
            env = Monitor(env, filename=str(monitor_dir / f"env_{rank}"))
        return env
    return _init


def make_vec_env(
    data: pd.DataFrame,
    n_envs: int,
    seed: int,
    monitor_dir: Optional[Path] = None,
    config: Optional[EnvConfig] = None,
) -> DummyVecEnv:
    """Build a vectorized env with `n_envs` parallel instances."""
    return DummyVecEnv([
        make_env(data, seed, i, monitor_dir, config) for i in range(n_envs)
    ])


# ============================================================
# PPO configuration
# ============================================================

def build_ppo_model(
    train_env: DummyVecEnv,
    log_dir: Path,
    seed: int,
) -> PPO:
    """Configure PPO with starting hyperparameters for small-state-space trading.

    Choices and rationale:
      learning_rate=3e-4    SB3 default. Conservative, works well for small MLPs.
      n_steps=2048          Rollout buffer per env per update. Total rollout
                            = n_envs * n_steps. With 4 envs => 8192 steps/update.
      batch_size=64         Minibatch size for the gradient update.
      n_epochs=10           Number of times we iterate over each rollout buffer.
      gamma=0.95            Per docs/00_mdp_formulation.md - calibrated for 4H
                            bars, gives ~3.3-day effective horizon.
      gae_lambda=0.95       Standard PPO default for advantage estimation.
      clip_range=0.2        Standard PPO clipping parameter.
      ent_coef=0.01         10x SB3 default. Encourages exploration; directly
                            addresses the policy-collapse failure mode from
                            the previous AlphaPulse RL attempt.
      vf_coef=0.5           Value function loss weighting (SB3 default).
      max_grad_norm=0.5     Gradient clipping (SB3 default).
      net_arch=[64, 64]     Two hidden layers of 64 units. Small, matched to
                            our 24-dim state. Bigger networks overfit on this
                            scale of data and aren't necessary for the policy
                            complexity our state can express.
    """
    policy_kwargs = {"net_arch": [64, 64]}

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        verbose=1,
        seed=seed,
    )
    return model


# ============================================================
# Run config persistence (for reproducibility)
# ============================================================

def save_run_config(
    model_dir: Path,
    args: argparse.Namespace,
    train_n: int,
    val_n: int,
) -> None:
    """Persist the exact configuration used. Lets you reproduce a run later."""
    config = {
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "data_start": args.data_start,
        "data_end": args.data_end,
        "val_fraction": args.val_fraction,
        "val_episode_length": args.val_episode_length,
        "train_bars": train_n,
        "val_bars": val_n,
        "checkpoint_freq": args.checkpoint_freq,
        "eval_freq": args.eval_freq,
        "run_id": args.run_id,
        "timestamp": datetime.now().isoformat(),
    }
    config_path = model_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Saved run config to %s", config_path)


# ============================================================
# Main training entrypoint
# ============================================================

def train(args: argparse.Namespace) -> None:
    """Top-level training orchestration."""
    # Auto-generate run ID if not provided
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_id = run_id

    log_dir = Path("runs") / run_id
    model_dir = Path("models") / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AlphaPulse PPO training")
    logger.info("=" * 60)
    logger.info("Run ID:       %s", run_id)
    logger.info("Log dir:      %s", log_dir)
    logger.info("Model dir:    %s", model_dir)
    logger.info("Timesteps:    %d", args.timesteps)
    logger.info("N envs:       %d", args.n_envs)
    logger.info("Seed:         %d", args.seed)
    logger.info("Data range:   %s -> %s", args.data_start, args.data_end)
    logger.info("Val fraction: %.2f", args.val_fraction)

    # 1. Load and split data
    train_data, val_data = load_and_split_data(
        start=args.data_start,
        end=args.data_end,
        val_fraction=args.val_fraction,
    )

    # Sanity: ensure both slices can construct their respective envs
    # Train uses default EnvConfig (episode 360); val uses shortened episode.
    train_min_bars = 600
    val_min_bars = 181 + args.val_episode_length + 1
    if len(train_data) < train_min_bars:
        raise ValueError(
            f"Train data too short: {len(train_data)} bars "
            f"(need >= {train_min_bars} for default EnvConfig)."
        )
    if len(val_data) < val_min_bars:
        raise ValueError(
            f"Val data too short: {len(val_data)} bars "
            f"(need >= {val_min_bars} for val_episode_length={args.val_episode_length})."
        )

    # 2. Save run config early (so a crash still leaves the metadata)
    save_run_config(model_dir, args, len(train_data), len(val_data))

    # 3. Build envs
    monitor_dir = log_dir / "monitor"
    train_env = make_vec_env(train_data, args.n_envs, args.seed, monitor_dir)

    # Val env uses a separate EnvConfig with shorter episodes so the smaller
    # val data slice can fit multiple evaluation episodes.
    val_config = EnvConfig(episode_length=args.val_episode_length)
    val_env = make_vec_env(
        val_data, n_envs=1, seed=args.seed + 1000, config=val_config,
    )

    # 4. Build the model
    model = build_ppo_model(train_env, log_dir, args.seed)
    logger.info("PPO model constructed. Policy:\n%s", model.policy)

    # 5. Callbacks
    # save_freq is per-env-step in vec_env; multiply by n_envs to get total-step semantics
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(model_dir),
        name_prefix="checkpoint",
        verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env=val_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    # 6. Train (catch Ctrl+C gracefully so partial runs are still saved)
    logger.info("Starting training for %d timesteps...", args.timesteps)
    start = datetime.now()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=False,
        )
        completed = True
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C). Saving current state...")
        completed = False

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(
        "Training %s in %.1f sec (%.0f fps avg)",
        "completed" if completed else "halted",
        elapsed,
        model.num_timesteps / max(elapsed, 1),
    )

    # 7. Save final model
    final_path = model_dir / "final.zip"
    model.save(str(final_path))
    logger.info("Saved final model to %s", final_path)

    # 8. Summary
    print("\n" + "=" * 60)
    print(f"Run complete: {run_id}")
    print(f"Trained timesteps: {model.num_timesteps}")
    print(f"Final model: {final_path}")
    print(f"Best model:  {model_dir / 'best' / 'best_model.zip'}")
    print(f"TensorBoard: tensorboard --logdir {log_dir.parent}")
    print("=" * 60)


# ============================================================
# CLI
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on BTC perpetual trading env",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS,
                        help="Number of parallel envs for rollouts")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--data-start", type=str, default=DEFAULT_DATA_START,
                        help="Training data start date (ISO)")
    parser.add_argument("--data-end", type=str, default=DEFAULT_DATA_END,
                        help="Training data end date (ISO)")
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
                        help="Fraction of data held out as val (chronologically last)")
    parser.add_argument("--val-episode-length", type=int, default=DEFAULT_VAL_EPISODE_LENGTH,
                        help="Episode length (in 4H bars) for val env")
    parser.add_argument("--checkpoint-freq", type=int, default=25_000,
                        help="Save checkpoint every N total timesteps")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="Evaluate on val env every N total timesteps")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Optional run ID (default: auto-generated timestamp)")
    return parser.parse_args(argv)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
