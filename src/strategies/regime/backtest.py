"""
src/strategies/regime/backtest.py - Regime-as-filter backtest (Level 2).

Level 1 (`detector.py`) validated that BTC 4H regimes are real, distinct, and
persistent. Level 2 asks the next question: are they TRADEABLE? Specifically, does
filtering market exposure by regime (be long in "favorable" regimes, hold cash
otherwise) improve risk-adjusted return versus buy-and-hold? This module runs that
backtest and prints a side-by-side comparison. It draws NO viability verdict - it
computes the numbers; interpretation happens elsewhere.

THE CARDINAL SIN THIS MODULE AVOIDS: look-ahead via in-sample regime labels.
Level 1 fits the HMM on the whole history and decodes the same history - fine for
characterization, fatal for a backtest, because bar t's label would embed
knowledge of bars after t. Here regimes are assigned WALK-FORWARD:

    * The HMM is refit on a cadence (`refit_every`) on an EXPANDING window of data
      strictly before the bars it will label. The scaler (z-score mean/std) is fit
      on the training window only and applied to later bars, so even the
      standardization carries no look-ahead.
    * Each bar t is labeled by Viterbi-decoding the observations from the current
      refit boundary THROUGH t (never past t) with the model trained on data before
      that boundary. Only data <= t determines bar t's label.
    * Regime count `n_states` (when auto-selected) is chosen ONCE from the first
      training window via AIC/BIC - a structural choice made only from past data,
      then held fixed. Never hardcoded.

DETECTION LAG: even walk-forward, you only CONFIRM a regime after the bar closes.
The strategy therefore acts on the PRIOR bar's regime: the position held over bar
t is decided from the regime known through bar t-1 (`favorable.shift(1)`). Never
same-bar. This is modeled explicitly in `_positions_from_labels`.

If look-ahead or lag were wrong, every performance number would be fiction; the
tests guard both above all else.

Generalization seams (for a future multi-asset regime scanner - built in now, not
a rewrite later):
    * asset/resolution fully parameterized via `RegimeBacktestConfig`;
    * per-asset AIC/BIC state selection when `n_states=None`;
    * `assign_regimes_walk_forward` is usable independently of the backtest (a
      scanner can get current regimes across assets without a full backtest);
    * `run_backtest` returns a structured, aggregatable `summary` dict.

CLI:
    python -m src.strategies.regime.backtest
    python -m src.strategies.regime.backtest --symbol PF_XBTUSD --cost-bps 10
    python -m src.strategies.regime.backtest --n-states 3 --refit-every 180
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.data.load_kraken import RESOLUTION_HOURS, build_dataset
from src.strategies.regime.detector import (
    HOURS_PER_YEAR,
    RegimeConfig,
    _apply_scaler,
    characterize_regimes,
    compute_features,
    fit_model,
    select_n_states,
)

logger = logging.getLogger(__name__)


# ============================================================
# Favorability rules (swappable)
# ============================================================
#
# A rule decides, from a regime's characterization AS KNOWN AT DECISION TIME
# (the trailing walk-forward fit, never full-sample), whether to be in the market
# ("favorable") or in cash. Each rule is (this_regime_stats, all_regime_stats) ->
# bool. Add a rule here and reference it by name in the config to swap behavior.

def _rule_positive_return_below_median_vol(
    regime_stats: dict[str, Any], all_stats: list[dict[str, Any]]
) -> bool:
    """Default: favorable = positive mean return AND below-median volatility.

    Captures the calm/uptrend regimes Level 1 found: drift up without the churn of
    the high-vol regimes. Median is taken across the CURRENT fit's regimes only.
    """
    vol_median = float(np.median(
        [r["volatility_profile"]["mean_realized_vol"] for r in all_stats]
    ))
    mean_ret = regime_stats["return_profile"]["mean_log_return"]
    mean_vol = regime_stats["volatility_profile"]["mean_realized_vol"]
    return bool(mean_ret > 0 and mean_vol <= vol_median)


def _rule_positive_return(regime_stats: dict[str, Any], all_stats: list[dict[str, Any]]) -> bool:
    """Simpler alternative: favorable = positive mean return (any volatility)."""
    return bool(regime_stats["return_profile"]["mean_log_return"] > 0)


FAVORABILITY_RULES: dict[str, Callable[[dict[str, Any], list[dict[str, Any]]], bool]] = {
    "positive_return_below_median_vol": _rule_positive_return_below_median_vol,
    "positive_return": _rule_positive_return,
}


# ============================================================
# Config
# ============================================================

@dataclass
class RegimeBacktestConfig:
    """Parameters for the regime-as-filter backtest.

    Attributes:
        symbol, resolution: asset + bar size (fully parameterized; nothing here is
            BTC-specific - running ETH is a config change).
        n_states: HMM state count; if None, auto-selected once via AIC/BIC from the
            first training window (never hardcoded to 4).
        refit_every: walk-forward refit cadence in bars (~30d at 4H).
        min_train_bars: minimum history before the first fit (need enough to
            identify regimes).
        transaction_cost_bps: per-side cost in bps, charged on every position change.
        favorable_regime_rule: key into FAVORABILITY_RULES.
        n_restarts, n_iter, random_seed, vol_window, trend_window, n_states_grid:
            passed through to the detector's RegimeConfig. Restarts/iters default
            lower than Level 1's because walk-forward refits many times; still
            deterministic via fixed seeds.
    """

    symbol: str = "PF_XBTUSD"
    resolution: str = "4h"
    n_states: int | None = None
    refit_every: int = 180
    min_train_bars: int = 1500
    transaction_cost_bps: float = 10.0
    favorable_regime_rule: str = "positive_return_below_median_vol"

    # Detector pass-through knobs.
    n_restarts: int = 5
    n_iter: int = 200
    random_seed: int = 42
    vol_window: int = 8
    trend_window: int = 12
    n_states_grid: tuple = (2, 3, 4)


def _regime_config(cfg: RegimeBacktestConfig) -> RegimeConfig:
    """Build the detector's RegimeConfig from the backtest config."""
    return RegimeConfig(
        symbol=cfg.symbol,
        resolution=cfg.resolution,
        n_states_grid=cfg.n_states_grid,
        n_iter=cfg.n_iter,
        n_restarts=cfg.n_restarts,
        random_seed=cfg.random_seed,
        vol_window=cfg.vol_window,
        trend_window=cfg.trend_window,
    )


def _favorability_map(char: dict[str, Any], rule_name: str) -> dict[int, bool]:
    """Map each state id -> favorable(bool) using the named rule and a fit's chars."""
    if rule_name not in FAVORABILITY_RULES:
        raise ValueError(f"Unknown favorable_regime_rule {rule_name!r}; "
                         f"known: {sorted(FAVORABILITY_RULES)}")
    rule = FAVORABILITY_RULES[rule_name]
    regimes = char["regimes"]
    all_stats = list(regimes.values())
    return {int(s): rule(regimes[s], all_stats) for s in regimes}


# ============================================================
# Walk-forward regime labeling (reusable by a future scanner)
# ============================================================

def assign_regimes_walk_forward(
    features: pd.DataFrame, config: RegimeBacktestConfig
) -> dict[str, Any]:
    """Assign a regime + favorability to each bar using ONLY its past.

    Expanding-window HMM, refit every `config.refit_every` bars starting at
    `config.min_train_bars`. Between refits, each bar t is labeled by decoding the
    observations from the refit boundary through t (standardized with the training
    scaler) and taking the last state - strictly causal (data <= t only).

    Returns:
        dict with:
          labels: DataFrame indexed by the labeled bars, columns
              [regime, favorable, fit_train_end] (fit_train_end = the boundary bar
              index whose fit labeled this row).
          n_refits: number of HMM refits performed.
          n_states: the (selected or configured) state count used.
    """
    rc = _regime_config(config)
    n = len(features)
    if n <= config.min_train_bars:
        raise ValueError(
            f"Not enough feature rows ({n}) for min_train_bars="
            f"{config.min_train_bars}; need more history."
        )

    # Structural choice: select n_states ONCE from the first training window (past
    # data only), then hold it fixed. Selecting per-asset satisfies the "no
    # hardcoded 4" requirement without re-selecting (and re-look-ahead-ing) each bar.
    first_train = features.iloc[: config.min_train_bars]
    if config.n_states is None:
        n_states = int(select_n_states(first_train, rc)["bic_preferred"])
        logger.info("Auto-selected n_states=%d (BIC, from first %d bars)",
                    n_states, config.min_train_bars)
    else:
        n_states = int(config.n_states)

    regime = np.full(n, -1, dtype=int)          # -1 = warmup / unlabeled
    favorable = np.zeros(n, dtype=bool)
    fit_train_end = np.full(n, -1, dtype=int)

    n_refits = 0
    i = config.min_train_bars
    while i < n:
        train = features.iloc[:i]  # strictly-before-i history
        fit = fit_model(train, n_states, rc)
        n_refits += 1
        model = fit["model"]
        mean, std = fit["scaler_mean"], fit["scaler_std"]

        # Favorability from THIS fit's characterization of its training window -
        # the regime properties known at decision time (not full-sample).
        train_states = model.predict(_apply_scaler(train, mean, std))
        char = characterize_regimes(train, train_states, train, rc,
                                    transition_matrix=model.transmat_)
        fav_map = _favorability_map(char, config.favorable_regime_rule)

        block_end = min(i + config.refit_every, n)
        for t in range(i, block_end):
            # Causal decode: standardize [i:t+1] with the TRAIN scaler, Viterbi,
            # take the last state. Only bars <= t enter this computation.
            window = features.iloc[i:t + 1]
            x_win = _apply_scaler(window, mean, std)
            s = int(model.predict(x_win)[-1])
            regime[t] = s
            favorable[t] = fav_map.get(s, False)
            fit_train_end[t] = i
        i = block_end

    mask = regime >= 0
    labels = pd.DataFrame(
        {"regime": regime[mask], "favorable": favorable[mask],
         "fit_train_end": fit_train_end[mask]},
        index=features.index[mask],
    )
    return {"labels": labels, "n_refits": n_refits, "n_states": n_states}


# ============================================================
# Position construction (detection lag lives here)
# ============================================================

def _positions_from_labels(favorable: pd.Series) -> pd.Series:
    """Long/flat target position with a ONE-BAR detection lag.

    The position held over bar t is decided from the regime known through bar t-1,
    so we shift the favorable flag forward by one bar. The first bar (no prior
    label) is flat. Returns 1.0 (long) / 0.0 (cash), indexed like `favorable`.
    """
    lagged = favorable.shift(1, fill_value=False)
    return lagged.astype(float)


# ============================================================
# Simulation engine (shared by filtered strategy AND buy-and-hold)
# ============================================================

def _simulate(
    positions: pd.Series, bar_returns: pd.Series, cost_bps: float
) -> dict[str, Any]:
    """Simulate an equity curve from per-bar target positions and bar returns.

    A position change from prev to `positions[t]` incurs |delta| * cost_bps/1e4 at
    bar t (prev starts at 0, so the first entry pays one side). Bar P&L =
    position[t] * bar_returns[t] - cost[t]. Buy-and-hold is just this with
    positions == 1 everywhere (one entry cost, then hold).

    Returns: equity (Series, starts at 1.0 before the first bar's return),
    strat_returns (Series), n_trades (int, position changes), turnover (float).
    """
    pos = positions.to_numpy(dtype=float)
    r = bar_returns.to_numpy(dtype=float)
    cost_rate = cost_bps / 1e4

    prev = 0.0
    equity = 1.0
    eq_curve = np.empty(len(pos))
    strat_ret = np.empty(len(pos))
    n_trades = 0
    turnover = 0.0
    for t in range(len(pos)):
        delta = abs(pos[t] - prev)
        if delta > 0:
            n_trades += 1
            turnover += delta
        cost = delta * cost_rate
        ret_t = pos[t] * r[t] - cost
        equity *= (1.0 + ret_t)
        eq_curve[t] = equity
        strat_ret[t] = ret_t
        prev = pos[t]

    return {
        "equity": pd.Series(eq_curve, index=positions.index),
        "strat_returns": pd.Series(strat_ret, index=positions.index),
        "n_trades": n_trades,
        "turnover": float(turnover),
    }


# ============================================================
# Performance metrics
# ============================================================

def _max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline of the equity curve (negative fraction)."""
    eq = equity.to_numpy(dtype=float)
    running_max = np.maximum.accumulate(eq)
    drawdown = eq / running_max - 1.0
    return float(drawdown.min()) if len(drawdown) else 0.0


def _compute_metrics(
    strat_returns: pd.Series, equity: pd.Series, periods_per_year: float
) -> dict[str, float]:
    """Standard performance metrics from a per-bar return series and equity curve.

    Sharpe uses a zero risk-free rate. Annualized return is geometric. Max drawdown
    is the most negative peak-to-trough fraction. Calmar = ann_return / |MDD|.
    """
    r = strat_returns.to_numpy(dtype=float)
    n = len(r)
    total_return = float(equity.iloc[-1] - 1.0) if n else 0.0
    if n > 0:
        annualized_return = float((1.0 + total_return) ** (periods_per_year / n) - 1.0)
    else:
        annualized_return = 0.0

    std = float(np.std(r))
    ann_vol = std * np.sqrt(periods_per_year)
    sharpe = float(np.mean(r) / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    mdd = _max_drawdown(equity)
    calmar = float(annualized_return / abs(mdd)) if mdd < 0 else float("nan")

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": calmar,
    }


# ============================================================
# Backtest engine
# ============================================================

def run_backtest(data: pd.DataFrame, config: RegimeBacktestConfig) -> dict[str, Any]:
    """Run the walk-forward regime-filter backtest vs buy-and-hold.

    Args:
        data: OHLCV DataFrame with a DatetimeIndex and a `close` column.
        config: backtest parameters.

    Returns:
        dict with:
          frame: per-bar DataFrame over the traded window (regime, favorable,
              position, bar_return, strat_return, equity_filtered, equity_bh);
          metrics: {"filtered": {...}, "buy_and_hold": {...}};
          summary: aggregatable dict (symbol, resolution, n_states_selected,
              current_regime, current_favorable, sharpe/mdd for both, uplift,
              time_in_market, n_trades, ...);
          n_refits: HMM refits performed.
    """
    bar_hours = RESOLUTION_HOURS[config.resolution]
    periods_per_year = HOURS_PER_YEAR / bar_hours

    features = compute_features(data, _regime_config(config))
    wf = assign_regimes_walk_forward(features, config)
    labels = wf["labels"]

    # Per-bar return realized OVER bar t = close_t / close_{t-1} - 1, aligned to the
    # labeled bars. Position over bar t uses the regime known through t-1 (lag).
    close = data["close"].astype(float)
    bar_return = (close / close.shift(1) - 1.0).reindex(labels.index)

    position = _positions_from_labels(labels["favorable"])

    # Trade only where we have both a defined lagged position AND a bar return.
    valid = bar_return.notna()
    position = position[valid]
    bar_return = bar_return[valid]

    filtered = _simulate(position, bar_return, config.transaction_cost_bps)
    # Buy-and-hold: constant unit position over the SAME window (one entry cost).
    bh_positions = pd.Series(1.0, index=position.index)
    buy_hold = _simulate(bh_positions, bar_return, config.transaction_cost_bps)

    m_filt = _compute_metrics(filtered["strat_returns"], filtered["equity"], periods_per_year)
    m_bh = _compute_metrics(buy_hold["strat_returns"], buy_hold["equity"], periods_per_year)

    time_in_market = float((position > 0).mean())
    frame = pd.DataFrame({
        "regime": labels["regime"].reindex(position.index),
        "favorable": labels["favorable"].reindex(position.index),
        "position": position,
        "bar_return": bar_return,
        "strat_return": filtered["strat_returns"],
        "equity_filtered": filtered["equity"],
        "equity_bh": buy_hold["equity"],
    })

    # Current (most recent) regime/favorability - what a scanner would surface live.
    current_regime = int(labels["regime"].iloc[-1])
    current_favorable = bool(labels["favorable"].iloc[-1])

    summary = {
        "symbol": config.symbol,
        "resolution": config.resolution,
        "n_states_selected": wf["n_states"],
        "n_refits": wf["n_refits"],
        "period_start": position.index[0],
        "period_end": position.index[-1],
        "n_bars": int(len(position)),
        "current_regime": current_regime,
        "current_favorable": current_favorable,
        "time_in_market": time_in_market,
        "n_trades": filtered["n_trades"],
        "turnover": filtered["turnover"],
        "sharpe_filtered": m_filt["sharpe"],
        "sharpe_bh": m_bh["sharpe"],
        "sharpe_uplift": (m_filt["sharpe"] - m_bh["sharpe"]),
        "max_drawdown_filtered": m_filt["max_drawdown"],
        "max_drawdown_bh": m_bh["max_drawdown"],
    }

    return {
        "frame": frame,
        "metrics": {"filtered": m_filt, "buy_and_hold": m_bh},
        "summary": summary,
        "n_refits": wf["n_refits"],
    }


# ============================================================
# Pretty printing
# ============================================================

def format_report(result: dict[str, Any], config: RegimeBacktestConfig) -> str:
    """Render the filtered-vs-buy-and-hold comparison table."""
    s = result["summary"]
    mf = result["metrics"]["filtered"]
    mb = result["metrics"]["buy_and_hold"]

    lines: list[str] = []
    bar = "=" * 70
    lines.append(bar)
    lines.append("  REGIME-AS-FILTER BACKTEST (Level 2)  -  walk-forward, cost-aware")
    lines.append(f"  {config.symbol} {config.resolution}  |  rule={config.favorable_regime_rule}")
    lines.append(bar)
    lines.append("")
    lines.append(f"  Period               : {s['period_start']}  ->  {s['period_end']}")
    lines.append(f"  Bars traded          : {s['n_bars']:,}")
    lines.append(f"  Regimes (selected)   : {s['n_states_selected']}  "
                 f"(HMM refits: {s['n_refits']})")
    lines.append(f"  Transaction cost     : {config.transaction_cost_bps:.1f} bps/side")
    lines.append(f"  Time in market       : {s['time_in_market'] * 100:.1f}%  "
                 f"(buy-and-hold: 100.0%)")
    lines.append(f"  Trades / turnover    : {s['n_trades']:,} / {s['turnover']:.1f}")
    lines.append(f"  Current regime       : {s['current_regime']}  "
                 f"({'FAVORABLE' if s['current_favorable'] else 'avoid'})")
    lines.append("")
    lines.append(f"  {'Metric':<22}{'Regime filter':>16}{'Buy & hold':>16}")
    lines.append("  " + "-" * 52)

    def row(label: str, a: float, b: float, pct: bool = False, dec: int = 3) -> str:
        if pct:
            av = f"{a * 100:+.2f}%"
            bv = f"{b * 100:+.2f}%"
        else:
            av = f"{a:+.{dec}f}"
            bv = f"{b:+.{dec}f}"
        return f"  {label:<22}{av:>16}{bv:>16}"

    lines.append(row("Total return", mf["total_return"], mb["total_return"], pct=True))
    lines.append(row("Annualized return", mf["annualized_return"], mb["annualized_return"], pct=True))
    lines.append(row("Annualized vol", mf["annualized_vol"], mb["annualized_vol"], pct=True))
    lines.append(row("Sharpe (rf=0)", mf["sharpe"], mb["sharpe"]))
    lines.append(row("Max drawdown", mf["max_drawdown"], mb["max_drawdown"], pct=True))
    lines.append(row("Calmar", mf["calmar"], mb["calmar"]))
    lines.append("")
    lines.append(f"  Sharpe uplift (filter - hold)   : {s['sharpe_uplift']:+.3f}")
    lines.append(f"  Drawdown change (filter - hold) : "
                 f"{(mf['max_drawdown'] - mb['max_drawdown']) * 100:+.2f} pp")
    lines.append("")
    lines.append(bar)
    lines.append("  Diagnostic only. No tradeability verdict is drawn here.")
    lines.append(bar)
    return "\n".join(lines)


# ============================================================
# CLI entrypoint
# ============================================================

def main(argv: list[str] | None = None) -> None:
    """Pull live OHLCV, run the walk-forward backtest, print the comparison."""
    parser = argparse.ArgumentParser(
        description="Regime-as-filter backtest (Level 2, walk-forward, diagnostic)."
    )
    parser.add_argument("--symbol", default="PF_XBTUSD")
    parser.add_argument("--resolution", default="4h")
    parser.add_argument("--n-states", type=int, default=None,
                        help="Force a regime count (default: auto-select via AIC/BIC).")
    parser.add_argument("--cost-bps", type=float, default=10.0,
                        help="Per-side transaction cost in bps (default: 10.0).")
    parser.add_argument("--refit-every", type=int, default=180,
                        help="Walk-forward refit cadence in bars (default: 180).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = RegimeBacktestConfig(
        symbol=args.symbol,
        resolution=args.resolution,
        n_states=args.n_states,
        transaction_cost_bps=args.cost_bps,
        refit_every=args.refit_every,
    )

    logger.info("Loading %s %s OHLCV ...", config.symbol, config.resolution)
    data = build_dataset(symbol=config.symbol, resolution=config.resolution)
    result = run_backtest(data, config)
    print(format_report(result, config))


if __name__ == "__main__":
    main()
