"""
src/strategies/regime/features.py - Causal regime posteriors as RL state features.

Level 3 feeds the market regime to the PPO agent as a state feature so it can learn
regime-conditional behavior (the direct fix for the original single-policy
overfitting), instead of us hand-coding a rule (Level 2). The representation is the
HMM's SOFT posterior over regimes (a probability vector per bar) - richer than a
hard label and smoother through transitions.

THE CARDINAL SIN, now inside the RL state: look-ahead. The posterior fed into bar
t's observation MUST use only data strictly before the agent acts on it. Two layers
guarantee this:

  1. CAUSAL walk-forward posteriors (reused from Level 2's machinery,
     `assign_regime_posteriors_walk_forward`): the HMM is refit on an expanding
     window, the scaler is fit on the training window only, and each bar's filtered
     posterior uses only observations up to and including that bar.
  2. ONE-BAR LAG: bar t's feature is the posterior known as of bar t-1. We shift the
     causal posterior sequence forward by one bar, so the agent never sees the
     current bar's regime before acting - the same detection-lag discipline Level 2
     applied to positions.

Warmup handling: bars before the first walk-forward fit (the first
`min_train_bars` feature rows), the feature-engineering warmup rows dropped by
`compute_features`, and the first labeled bar (which has no t-1 posterior) cannot
have a causal regime. They receive a NEUTRAL uniform posterior (1/n_states each) -
never a look-ahead-contaminated value. This is documented and intentional.

The returned array is aligned POSITIONALLY to the full input `data` (one row per
bar), ready to hand to `TradingEnv(regime_features=...)`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.strategies.regime.backtest import (
    RegimeBacktestConfig,
    assign_regime_posteriors_walk_forward,
)
from src.strategies.regime.detector import RegimeConfig, compute_features

logger = logging.getLogger(__name__)


def build_causal_regime_posteriors(
    data: pd.DataFrame, config: RegimeBacktestConfig
) -> tuple[np.ndarray, int]:
    """Build per-bar causal, one-bar-lagged regime posteriors aligned to `data`.

    Args:
        data: OHLCV DataFrame with a DatetimeIndex and a `close` column (the full
            dataset; split it AFTER building so train/val posteriors stay aligned).
        config: Level 2 backtest config (reused) controlling the walk-forward -
            symbol/resolution, refit cadence, min_train_bars, n_states (or None to
            auto-select via AIC/BIC), and the detector pass-through knobs.

    Returns:
        (posteriors, n_states) where posteriors has shape [len(data), n_states];
        each row is a probability distribution over regimes. Rows without a causal
        regime (warmup / first labeled bar) are uniform (1/n_states).
    """
    rc = RegimeConfig(
        symbol=config.symbol, resolution=config.resolution,
        vol_window=config.vol_window, trend_window=config.trend_window,
    )
    # Features carry their own warmup: compute_features drops the first
    # max(vol_window, trend_window) rows, so `features.index` is a tail subset of
    # `data.index`. Walk-forward then labels only bars >= min_train_bars within it.
    features = compute_features(data, rc)

    wf = assign_regime_posteriors_walk_forward(features, config)
    posteriors = wf["posteriors"]          # DataFrame over LABELED feature bars
    n_states = wf["n_states"]
    uniform = np.full(n_states, 1.0 / n_states, dtype=np.float32)

    # 1) Causal posterior per FEATURE bar (uniform where unlabeled = warmup).
    n_feat = len(features)
    feat_post = np.tile(uniform, (n_feat, 1))
    labeled_pos = features.index.get_indexer(posteriors.index)
    feat_post[labeled_pos] = posteriors.to_numpy(dtype=np.float32)

    # 2) One-bar lag along the feature axis: bar t's feature = posterior as of t-1.
    #    The first feature bar has no predecessor -> uniform. Shifting the whole
    #    array (warmup included) also makes the FIRST labeled bar inherit uniform,
    #    exactly as intended (no t-1 posterior exists for it yet).
    feat_post_lagged = np.empty_like(feat_post)
    feat_post_lagged[0] = uniform
    feat_post_lagged[1:] = feat_post[:-1]

    # 3) Expand to the full data index; bars not in the feature frame (dropped
    #    feature-warmup rows) are uniform.
    full = np.tile(uniform, (len(data), 1))
    data_pos = data.index.get_indexer(features.index)
    full[data_pos] = feat_post_lagged

    # Count rows that carry a genuine (non-uniform) causal posterior vs the
    # neutral warmup fill, so the log reflects reality rather than feature coverage.
    n_causal = int((~np.all(np.isclose(full, uniform), axis=1)).sum())
    logger.info(
        "Built causal regime posteriors: n_states=%d, %d/%d data bars carry a "
        "lagged causal posterior (rest uniform warmup), refits=%d.",
        n_states, n_causal, len(data), wf["n_refits"],
    )
    return full.astype(np.float32), n_states
