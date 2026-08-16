"""
src/strategies/regime/detector.py - Market regime detection (Level 1 diagnostic).

Fits a Gaussian Hidden Markov Model to multi-year BTC 4H data to discover latent
market regimes, then characterizes them. The question this answers - BEFORE any
regime-switching machinery is built - is whether BTC regimes are distinct,
persistent, and actionable. It is diagnostic only: it discovers regimes, computes
their statistics, and prints them. It contains no trading logic and draws no
viability conclusion.

Why regimes: a prior finding was that the RL agent overfit partly because it tried
to learn ONE policy across all market conditions. If the data cleanly separates
into a few persistent regimes with distinct return/vol profiles, a regime-aware
approach is worth building; if regimes are fleeting or indistinguishable, it is
not. This module measures that.

Features the HMM observes (kept deliberately small - HMM regimes are only as
interpretable as their features):
    log_return    log(close_t / close_{t-1})                        - direction
    realized_vol  rolling std of log returns over `vol_window` bars  - volatility
    trend         (close_t - SMA_n) / SMA_n over `trend_window` bars - trend strength

NO LOOK-AHEAD: every feature at bar t uses only closes at or before t.

Known simplification (documented, acceptable for a diagnostic): features are
z-scored on the FULL sample before fitting. A deployed model would standardize
with rolling/expanding statistics so the scaling itself carries no look-ahead;
here we accept full-sample z-scoring because nothing is traded on the output.

HMM state labels (0, 1, 2, ...) are arbitrary - a regime is defined by its
computed properties (vol, return), not its index. Characterization therefore
describes regimes by their statistics and attaches a descriptive label.

CLI:
    python -m src.strategies.regime.detector
    python -m src.strategies.regime.detector --n-states 3
    python -m src.strategies.regime.detector --resolution 4h
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.load_kraken import RESOLUTION_HOURS, build_dataset
# Reuse the env's realized-vol helper (rolling std of log returns) rather than
# re-implementing it; it is cleanly importable and identical to what we need.
from src.env.trading_env import _realized_vol

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

HOURS_PER_YEAR = 24 * 365  # 8760; for annualizing per-bar stats


# ============================================================
# Config
# ============================================================

@dataclass
class RegimeConfig:
    """Parameters for HMM regime detection.

    Attributes:
        symbol: Kraken Futures perpetual symbol.
        resolution: bar resolution (must be a key of RESOLUTION_HOURS).
        n_states_grid: HMM state counts to compare via AIC/BIC.
        covariance_type: emission covariance structure ("full" is standard for
            regime HMMs - each state gets its own full covariance matrix).
        n_iter: EM (Baum-Welch) max iterations per fit.
        n_restarts: random restarts; the best log-likelihood fit is kept (HMMs are
            sensitive to initialization, so multiple restarts mitigate local optima).
        random_seed: base seed; restart i uses random_seed + i (reproducible).
        vol_window: bars for realized-volatility rolling std (8 * 4h = 32h).
        trend_window: bars for the trend moving average (12 * 4h = 48h ~ 2 days).
    """

    symbol: str = "PF_XBTUSD"
    resolution: str = "4h"
    n_states_grid: tuple = (2, 3, 4)
    covariance_type: str = "full"
    n_iter: int = 1000
    n_restarts: int = 10
    random_seed: int = 42
    vol_window: int = 8
    trend_window: int = 12


# ============================================================
# Feature engineering
# ============================================================

def compute_features(ohlcv: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    """Compute the RAW (un-standardized) regime features from OHLCV.

    Every feature at bar t uses only data at or before t (no look-ahead). Warmup
    rows where any feature is undefined are dropped.

    Args:
        ohlcv: DataFrame with a DatetimeIndex and at least a `close` column.
        config: regime parameters (vol_window, trend_window).

    Returns:
        DataFrame indexed by the surviving bars with columns
        [log_return, realized_vol, trend]. Raw units - standardization happens
        only inside the fitting path, so these stay interpretable for
        characterization.
    """
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv must have a 'close' column")
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise ValueError("ohlcv must have a DatetimeIndex")

    close = ohlcv["close"].astype(float)

    # Direction: per-bar log return. log_return_t uses close_t and close_{t-1}.
    log_return = np.log(close / close.shift(1))

    # Volatility: rolling std of log returns over `vol_window` bars (<= t).
    realized_vol = _realized_vol(log_return, config.vol_window)

    # Trend strength: normalized distance of close from its trailing SMA. Positive
    # => price above its recent average (up-trend), negative => below. The SMA at t
    # averages closes at or before t, so no look-ahead. Chosen over a slope fit for
    # directness and unit-free comparability across price levels.
    sma = close.rolling(config.trend_window).mean()
    trend = (close - sma) / sma

    features = pd.DataFrame(
        {"log_return": log_return, "realized_vol": realized_vol, "trend": trend}
    )
    # Drop warmup rows (first bar for log_return, vol_window for vol, trend_window
    # for the SMA) where any feature is NaN.
    features = features.dropna()
    return features


def _standardize(features: pd.DataFrame) -> np.ndarray:
    """Z-score each feature column (full-sample; see module docstring caveat)."""
    x = features.to_numpy(dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 0, std, 1.0)  # avoid divide-by-zero on a flat column
    return (x - mean) / std


# ============================================================
# Information criteria
# ============================================================

def _hmm_n_params(n_states: int, n_features: int, covariance_type: str) -> int:
    """Number of free parameters in a GaussianHMM.

        startprob:   n_states - 1                    (simplex, sums to 1)
        transmat:    n_states * (n_states - 1)       (each row a simplex)
        means:       n_states * n_features
        covariances: depends on covariance_type; "full" => a symmetric
                     n_features x n_features matrix per state =
                     n_features * (n_features + 1) / 2 free entries each.
    """
    start = n_states - 1
    trans = n_states * (n_states - 1)
    means = n_states * n_features
    if covariance_type == "full":
        cov_per_state = n_features * (n_features + 1) // 2
    elif covariance_type == "diag":
        cov_per_state = n_features
    elif covariance_type == "spherical":
        cov_per_state = 1
    elif covariance_type == "tied":
        # One shared full covariance across states.
        return start + trans + means + n_features * (n_features + 1) // 2
    else:
        raise ValueError(f"Unknown covariance_type {covariance_type!r}")
    return start + trans + means + n_states * cov_per_state


def _aic_bic(log_likelihood: float, n_params: int, n_obs: int) -> tuple[float, float]:
    """AIC and BIC from total log-likelihood, parameter count, and #observations.

        AIC = -2*LL + 2*p
        BIC = -2*LL + p*ln(N)
    Lower is better for both.
    """
    aic = -2.0 * log_likelihood + 2.0 * n_params
    bic = -2.0 * log_likelihood + n_params * np.log(n_obs)
    return float(aic), float(bic)


# ============================================================
# Fitting
# ============================================================

def fit_regimes(features: pd.DataFrame, n_states: int, config: RegimeConfig) -> dict[str, Any]:
    """Fit a GaussianHMM with `n_states` and decode the regime sequence.

    Runs `config.n_restarts` fits from different seeds and keeps the one with the
    highest log-likelihood (HMMs are init-sensitive). Features are z-scored before
    fitting (Gaussian emissions are scale-sensitive).

    Args:
        features: raw feature DataFrame from `compute_features`.
        n_states: number of HMM hidden states.
        config: regime parameters.

    Returns:
        dict with keys: model (fitted GaussianHMM), states (np.ndarray of decoded
        Viterbi states aligned positionally to `features`), index (the feature
        DatetimeIndex), log_likelihood, n_params, aic, bic, n_states, converged.
    """
    from hmmlearn.hmm import GaussianHMM  # local import: heavy optional dependency

    x = _standardize(features)
    n_obs, n_features = x.shape

    best_model = None
    best_ll = -np.inf
    for i in range(config.n_restarts):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=config.covariance_type,
            n_iter=config.n_iter,
            random_state=config.random_seed + i,  # deterministic per restart
        )
        try:
            model.fit(x)
            ll = model.score(x)
        except Exception as e:  # a degenerate restart shouldn't kill the sweep
            logger.warning("HMM restart %d (n_states=%d) failed: %s", i, n_states, e)
            continue
        if ll > best_ll:
            best_ll, best_model = ll, model

    if best_model is None:
        raise RuntimeError(f"All {config.n_restarts} HMM restarts failed for n_states={n_states}")

    states = best_model.predict(x)  # Viterbi most-likely state per bar
    n_params = _hmm_n_params(n_states, n_features, config.covariance_type)
    aic, bic = _aic_bic(best_ll, n_params, n_obs)

    return {
        "model": best_model,
        "states": states,
        "index": features.index,
        "log_likelihood": float(best_ll),
        "n_params": n_params,
        "aic": aic,
        "bic": bic,
        "n_states": n_states,
        "converged": bool(best_model.monitor_.converged),
    }


def select_n_states(features: pd.DataFrame, config: RegimeConfig) -> dict[str, Any]:
    """Fit for each n_states in the grid and report AIC/BIC for comparison.

    Does NOT hard-pick a winner - it reports every candidate's criteria (and, for
    convenience, which n_states each criterion would prefer) so interpretation can
    decide. Lower AIC/BIC is better.

    Returns:
        dict with "per_n_states" (n_states -> {log_likelihood, n_params, aic, bic,
        converged}), "aic_preferred", and "bic_preferred".
    """
    per: dict[int, dict[str, Any]] = {}
    for n_states in config.n_states_grid:
        fit = fit_regimes(features, n_states, config)
        per[n_states] = {
            "log_likelihood": fit["log_likelihood"],
            "n_params": fit["n_params"],
            "aic": fit["aic"],
            "bic": fit["bic"],
            "converged": fit["converged"],
        }
        logger.info(
            "n_states=%d: LL=%.1f AIC=%.1f BIC=%.1f (converged=%s)",
            n_states, fit["log_likelihood"], fit["aic"], fit["bic"], fit["converged"],
        )

    aic_pref = min(per, key=lambda k: per[k]["aic"])
    bic_pref = min(per, key=lambda k: per[k]["bic"])
    return {"per_n_states": per, "aic_preferred": aic_pref, "bic_preferred": bic_pref}


# ============================================================
# Characterization (the key output)
# ============================================================

def _run_lengths(state_sequence: np.ndarray) -> list[tuple[int, int]]:
    """Return (state, run_length) for each maximal run of identical states."""
    runs: list[tuple[int, int]] = []
    if len(state_sequence) == 0:
        return runs
    cur = state_sequence[0]
    length = 1
    for s in state_sequence[1:]:
        if s == cur:
            length += 1
        else:
            runs.append((int(cur), length))
            cur, length = s, 1
    runs.append((int(cur), length))
    return runs


def _empirical_transition_matrix(state_sequence: np.ndarray, n_states: int) -> np.ndarray:
    """Row-stochastic transition matrix estimated by counting t -> t+1 moves."""
    counts = np.zeros((n_states, n_states), dtype=float)
    for a, b in zip(state_sequence[:-1], state_sequence[1:]):
        counts[int(a), int(b)] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    # A state with no outgoing observations keeps a zero row (rather than fabricate).
    return np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)


def _regime_label(mean_return: float, mean_vol: float, vol_median: float,
                  ret_scale: float) -> str:
    """Descriptive label from a regime's stats (labels, not state indices, matter).

    Volatility is called high/low relative to the median regime vol; direction is
    up/down/choppy by the sign and size of mean return vs a small scale threshold.
    """
    vol_tag = "high-vol" if mean_vol >= vol_median else "low-vol"
    # "choppy" when the mean per-bar return is small relative to the cross-regime
    # spread of returns; otherwise up/down-trend by sign.
    if abs(mean_return) < 0.2 * ret_scale or ret_scale == 0:
        dir_tag = "choppy"
    else:
        dir_tag = "uptrend" if mean_return > 0 else "downtrend"
    return f"{vol_tag}/{dir_tag}"


def characterize_regimes(
    features: pd.DataFrame,
    state_sequence: np.ndarray,
    ohlcv: pd.DataFrame,
    config: RegimeConfig,
    transition_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute per-regime statistics - the key diagnostic output.

    Args:
        features: raw features (must include log_return, realized_vol), aligned
            positionally to `state_sequence`.
        state_sequence: decoded state per bar (same length/order as `features`).
        ohlcv: source OHLCV (unused directly here beyond context; features carry
            the return/vol already, kept for signature/extension parity).
        config: regime parameters (resolution drives bar->time conversion).
        transition_matrix: optional HMM transition matrix (`model.transmat_`) to
            report. If None, an empirical matrix is estimated from the sequence.

    Returns:
        dict with per-regime frequency, persistence (bars/hours/days), return,
        volatility and directionality profiles, the transition matrix, and a
        cross-regime distinctness summary.
    """
    if len(state_sequence) != len(features):
        raise ValueError("state_sequence and features must have equal length")

    bar_hours = RESOLUTION_HOURS[config.resolution]
    periods_per_year = HOURS_PER_YEAR / bar_hours  # e.g. 8760/4 = 2190 for 4H

    log_ret = features["log_return"].to_numpy(dtype=float)
    vol = features["realized_vol"].to_numpy(dtype=float)
    states = np.asarray(state_sequence)
    n_total = len(states)
    unique_states = sorted(int(s) for s in np.unique(states))
    n_states = len(unique_states)

    # Persistence: run lengths grouped by state.
    runs = _run_lengths(states)
    runs_by_state: dict[int, list[int]] = {s: [] for s in unique_states}
    for s, length in runs:
        runs_by_state[s].append(length)

    # Cross-regime scale for labeling: spread of per-regime mean returns.
    prelim_means = {s: float(np.mean(log_ret[states == s])) for s in unique_states}
    ret_scale = max(prelim_means.values()) - min(prelim_means.values()) if n_states > 1 else 0.0
    vol_means = {s: float(np.mean(vol[states == s])) for s in unique_states}
    vol_median = float(np.median(list(vol_means.values())))

    regimes: dict[int, dict[str, Any]] = {}
    for s in unique_states:
        mask = states == s
        r = log_ret[mask]
        v = vol[mask]
        durations = np.array(runs_by_state[s], dtype=float)

        mean_ret = float(np.mean(r))
        std_ret = float(np.std(r))
        # Annualized Sharpe-like ratio: per-bar mean/std scaled by sqrt(bars/yr).
        ann_sharpe = (
            float(mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else float("nan")
        )
        mean_dur_bars = float(np.mean(durations))

        regimes[s] = {
            "label": _regime_label(mean_ret, vol_means[s], vol_median, ret_scale),
            "frequency": float(mask.sum() / n_total),
            "n_bars": int(mask.sum()),
            "persistence": {
                "mean_bars": mean_dur_bars,
                "median_bars": float(np.median(durations)),
                "max_bars": float(np.max(durations)),
                "mean_hours": mean_dur_bars * bar_hours,
                "mean_days": mean_dur_bars * bar_hours / 24.0,
                "n_episodes": int(len(durations)),
            },
            "return_profile": {
                "mean_log_return": mean_ret,
                "std_log_return": std_ret,
                "annualized_sharpe": ann_sharpe,
            },
            "volatility_profile": {
                "mean_realized_vol": vol_means[s],
            },
            "directionality": {
                "frac_positive": float(np.mean(r > 0)),
                "frac_negative": float(np.mean(r < 0)),
            },
        }

    # Transition matrix: prefer the HMM's own if provided, else empirical.
    if transition_matrix is not None:
        tmat = np.asarray(transition_matrix, dtype=float)
        tmat_source = "hmm"
    else:
        tmat = _empirical_transition_matrix(states, max(unique_states) + 1)
        tmat_source = "empirical"

    # Distinctness: do regimes actually differ? Report the spread (max - min) of
    # per-regime mean return and mean vol across regimes.
    mean_returns = [regimes[s]["return_profile"]["mean_log_return"] for s in unique_states]
    mean_vols = [regimes[s]["volatility_profile"]["mean_realized_vol"] for s in unique_states]
    distinctness = {
        "return_spread": float(max(mean_returns) - min(mean_returns)) if n_states > 1 else 0.0,
        "vol_spread": float(max(mean_vols) - min(mean_vols)) if n_states > 1 else 0.0,
    }

    return {
        "n_states": n_states,
        "regimes": regimes,
        "transition_matrix": tmat.tolist(),
        "transition_matrix_source": tmat_source,
        "distinctness": distinctness,
    }


# ============================================================
# Pretty printing
# ============================================================

def format_selection(selection: dict[str, Any]) -> str:
    """Render the AIC/BIC model-selection comparison."""
    lines: list[str] = []
    bar = "=" * 68
    lines.append(bar)
    lines.append("  HMM MODEL SELECTION  (lower AIC/BIC is better)")
    lines.append(bar)
    lines.append(f"  {'n_states':>8}  {'log-lik':>12}  {'params':>7}  {'AIC':>12}  {'BIC':>12}  {'conv':>5}")
    lines.append("  " + "-" * 64)
    per = selection["per_n_states"]
    for n in sorted(per):
        row = per[n]
        lines.append(
            f"  {n:>8}  {row['log_likelihood']:>12.1f}  {row['n_params']:>7}"
            f"  {row['aic']:>12.1f}  {row['bic']:>12.1f}  {str(row['converged']):>5}"
        )
    lines.append("  " + "-" * 64)
    lines.append(f"  AIC prefers n_states={selection['aic_preferred']}; "
                 f"BIC prefers n_states={selection['bic_preferred']}")
    return "\n".join(lines)


def format_characterization(char: dict[str, Any], config: RegimeConfig) -> str:
    """Render per-regime characterization + transition matrix + distinctness."""
    lines: list[str] = []
    bar = "=" * 68
    lines.append(bar)
    lines.append(f"  REGIME CHARACTERIZATION  ({char['n_states']} states, "
                 f"{config.symbol} {config.resolution})")
    lines.append(bar)

    regimes = char["regimes"]
    for s in sorted(regimes):
        r = regimes[s]
        p = r["persistence"]
        rp = r["return_profile"]
        lines.append("")
        lines.append(f"Regime {s}: {r['label']}")
        lines.append("-" * 68)
        lines.append(f"  Frequency            : {r['frequency'] * 100:6.2f}%  ({r['n_bars']:,} bars)")
        lines.append(f"  Persistence (mean)   : {p['mean_bars']:6.2f} bars"
                     f"  ({p['mean_hours']:,.1f}h / {p['mean_days']:.2f}d)")
        lines.append(f"  Persistence med/max  : {p['median_bars']:.1f} / {p['max_bars']:.0f} bars"
                     f"  ({p['n_episodes']} episodes)")
        lines.append(f"  Mean log return/bar  : {rp['mean_log_return']:+.6f}"
                     f"  (std {rp['std_log_return']:.6f})")
        lines.append(f"  Annualized Sharpe    : {rp['annualized_sharpe']:+.3f}")
        lines.append(f"  Mean realized vol    : {r['volatility_profile']['mean_realized_vol']:.6f}")
        lines.append(f"  Directionality       : {r['directionality']['frac_positive'] * 100:5.1f}% up / "
                     f"{r['directionality']['frac_negative'] * 100:5.1f}% down")

    lines.append("")
    lines.append(f"Transition matrix ({char['transition_matrix_source']}; "
                 f"diagonal = P(stay), rows sum to 1)")
    lines.append("-" * 68)
    tmat = char["transition_matrix"]
    header = "         " + "".join(f"->{j:>7}" for j in range(len(tmat)))
    lines.append(header)
    for i, rowvals in enumerate(tmat):
        cells = "".join(f"{v:>9.3f}" for v in rowvals)
        lines.append(f"  from {i:>2}{cells}")

    lines.append("")
    lines.append("Regime distinctness (spread across regimes; larger = more distinct)")
    lines.append("-" * 68)
    d = char["distinctness"]
    lines.append(f"  Mean-return spread   : {d['return_spread']:.6f}  (max - min per-bar mean)")
    lines.append(f"  Mean-vol spread      : {d['vol_spread']:.6f}")

    lines.append("")
    lines.append(bar)
    lines.append("  Diagnostic only. No tradeability conclusion is drawn here.")
    lines.append(bar)
    return "\n".join(lines)


# ============================================================
# CLI entrypoint
# ============================================================

def main(argv: list[str] | None = None) -> None:
    """Pull live OHLCV, compare state counts, and characterize regimes."""
    parser = argparse.ArgumentParser(
        description="Market regime detection (Level 1 HMM diagnostic)."
    )
    parser.add_argument("--symbol", default="PF_XBTUSD",
                        help="Kraken Futures symbol (default: PF_XBTUSD).")
    parser.add_argument("--resolution", default="4h",
                        help="Bar resolution (default: 4h).")
    parser.add_argument("--n-states", type=int, default=None,
                        help="Force a specific state count for characterization "
                             "(default: use the BIC-preferred count).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = RegimeConfig(symbol=args.symbol, resolution=args.resolution)

    logger.info("Loading %s %s OHLCV ...", config.symbol, config.resolution)
    # build_dataset returns OHLCV + funding; we only need OHLCV columns.
    data = build_dataset(symbol=config.symbol, resolution=config.resolution)
    features = compute_features(data, config)
    logger.info("Computed %d feature rows (after warmup).", len(features))

    # Model selection across the grid.
    selection = select_n_states(features, config)
    print(format_selection(selection))

    # Characterize at the chosen count (BIC-preferred unless forced).
    chosen = args.n_states if args.n_states is not None else selection["bic_preferred"]
    print(f"\nCharacterizing at n_states={chosen} "
          f"({'forced' if args.n_states is not None else 'BIC-preferred'}) ...\n")
    fit = fit_regimes(features, chosen, config)
    char = characterize_regimes(
        features, fit["states"], data, config,
        transition_matrix=fit["model"].transmat_,
    )
    print(format_characterization(char, config))


if __name__ == "__main__":
    main()
