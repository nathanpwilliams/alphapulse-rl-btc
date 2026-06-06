# Phase 0 — MDP Formulation: BTC-USD Trading Agent

| | |
|---|---|
| **Project** | AlphaPulse RL Trading Bot (v1) |
| **Author** | Nathan |
| **Status** | Draft v1.1 (post-decision lock) |
| **Date** | 2026-05-29 |
| **Phase** | 0 — Formulation |
| **Next Phase** | 1 — Backtest training |
| **Execution venue** | Kraken Futures testnet (paper), spot perp BTCUSD |

---

## 1. Purpose

This document specifies the reinforcement learning problem **before any code is written**. Every design choice below is a decision I will need to defend in postmortems. If it isn't in this document, it isn't decided yet — and decisions made implicitly during coding are the ones that produce collapsed models.

The previous RL attempt (`alpha_rl_aggressive_20250905_011903`) skipped this phase. The result was a PPO model whose policy collapsed during training and had to be wrapped in a hand-coded heuristic to function. This is the most common failure mode in applied RL, and it is preventable by writing the MDP down clearly *first*.

---

## 2. Problem Statement

> **Learn a policy π(a|s) that maps observed market state to a target position size in BTC-USD, such that the expected discounted cumulative log-return of the portfolio is maximized over an episode, subject to transaction costs.**

This is a sequential decision problem with the following hard characteristics:

- **Non-stationary environment.** Crypto market regimes shift on timescales (weeks to months) shorter than typical RL training horizons.
- **Low signal-to-noise ratio.** Most hourly returns are dominated by noise, not predictable patterns.
- **Partial observability.** The true state (e.g., order book imbalance, off-exchange flow, macro positioning) is not observable from price data alone. We approximate the state with features.
- **Sparse, delayed reward.** A correct decision now may only pay off many steps later.

These properties violate several standard RL assumptions and are the reason most RL-for-trading research fails to outperform simple baselines after costs. The point of this project is **not to disprove this**; it is to engage with these challenges honestly and learn what RL can and cannot contribute.

---

## 3. The Markov Decision Process

Formally, we model trading as an MDP $\langle \mathcal{S}, \mathcal{A}, P, r, \gamma \rangle$:

- $\mathcal{S}$: state space (market + portfolio observations)
- $\mathcal{A}$: action space (target position sizes)
- $P(s' \mid s, a)$: transition dynamics — **unknown** (this is why we use model-free RL)
- $r(s, a, s')$: reward function (defined below)
- $\gamma$: discount factor

### 3.1 State Space $\mathcal{S}$

The state at time $t$ is a vector $s_t \in \mathbb{R}^{d}$ composed of three groups:

**(a) Market features (12 + 5 = 17 dims).** All derived from data available *strictly before time t* (no look-ahead).

| Feature | Dim | Description | Normalization |
|---|---|---|---|
| Lagged 4H log-returns | 12 | $\log(p_{t-k}/p_{t-k-1})$ for $k = 1..12$ (≈2 days of context) | None (already scale-free) |
| RSI(14) | 1 | Relative Strength Index on 4H bars | Divided by 100 → [0,1] |
| MACD histogram | 1 | (MACD line − signal line) on 4H | Divided by current price |
| ATR(14) | 1 | Average True Range on 4H | Divided by current price → relative vol |
| Bollinger %B | 1 | (price − lower band) / (upper − lower band) | Already in ~[0,1] |
| Realized vol (8 bars ≈ 32h) | 1 | std of last 8 4H log-returns | Standardized vs trailing 30d |

**(b) Portfolio features (3 dims).** Internal state the agent needs to make decisions:

| Feature | Description |
|---|---|
| Current position | Signed position as fraction of capital, $\in [-1, 1]$ (negative = short) |
| Unrealized P&L | Position-adjusted: `sign(pos) × (current_price − entry_price) / entry_price`, or 0 if flat |
| Steps in position | Number of 4H bars since position opened/last flipped, log-scaled |

**(c) Calendar features (4 dims, optional).** Sin/cos encodings of bar-of-day (4H bars → 6 per day) and day-of-week. Crypto trades 24/7 but funding-rate dynamics and global session boundaries still create patterns.

**Total state dimension: ~24.**

#### 3.1.1 Why this state, and not something richer?

I considered and **rejected** the following for v1:

- **Order book features** (depth, imbalance). Require L2 data, costly to source and store, and add infrastructure burden before I know if the basic version works.
- **Sentiment / on-chain features.** Same logic — defer until v1 is converged.
- **Multi-asset features** (BTC/ETH ratio, correlation regime). Out of scope; v1 is single-asset.
- **Deep encoders (LSTM, Transformer) over raw OHLCV.** Adds substantial training instability and obscures debugging. The agent can always be re-trained with a deeper encoder once the simple version is understood.

**Principle: start with the smallest state that can plausibly work. Add complexity only after demonstrating that the simple version fails for understood reasons.**

### 3.2 Action Space $\mathcal{A}$

$$ a_t \in [-1, 1] $$

The action is a **continuous signed target position size** expressed as a fraction of capital, executed against Kraken Futures perpetual BTCUSD on the testnet. Negative values represent short positions.

**Concretely:** if current position is +0.4 (40% long) and $a_t = -0.3$ (30% short), the env closes the long, opens a short, and pays transaction costs on the total notional moved (0.7 units of capital crossed).

#### 3.2.1 Design decisions

- **Continuous, not discrete (buy/sell/hold).** Discrete actions lose information and produce noisier policy gradients. PPO handles continuous action spaces natively via a Gaussian policy head.
- **Target position semantics, not delta semantics.** Modelling actions as target positions rather than buy/sell-N-contracts makes the policy invariant to current position size, which simplifies learning.
- **Long-short ([-1, 1]).** Kraken Futures testnet supports BTC perps which allow shorting. This doubles the policy's expressive range vs long-only and lets the agent profit in bear regimes — necessary for an honest test of whether RL captures regime-conditional behavior.
- **No leverage in v1.** Position size is capped at $|a| \leq 1$ (100% notional, not leveraged). The exchange supports leverage but adding it at v1 increases the blow-up risk during training exploration.
- **Position-flip costs are real.** Going from +1 → −1 incurs cost on a 2.0 notional swing. The reward function (§3.3) charges this explicitly so the agent learns conviction, not whipsaw.

### 3.3 Reward Function

$$ r(s_t, a_t, s_{t+1}) = \log\left(\frac{V_{t+1}}{V_t}\right) - c \cdot |a_t - a_{t-1}| - f_t \cdot a_t \cdot \Delta_t $$

Where:
- $V_t$ = portfolio value at time $t$ (cash + position × price, including unrealized P&L on perp)
- $c$ = transaction cost rate (0.0005 = 5 bps per side for Kraken Futures taker)
- $|a_t - a_{t-1}|$ = magnitude of the rebalance (notional crossed)
- $f_t$ = funding rate for the period (positive = longs pay shorts)
- $a_t \cdot \Delta_t$ = position × funding interval (3 4H bars per 8h funding period, simplified to per-bar attribution)

The funding term means: holding a long position when funding is positive reduces reward; holding a short when funding is positive increases reward. This is a real signal in crypto perps that a long-short agent can learn to exploit — perpetual funding regimes carry information about market positioning.

#### 3.3.1 Why log returns?

- **Time-additive.** Sum of step log-returns equals log of total return. The agent's discounted return is therefore a meaningful proxy for terminal log-wealth.
- **Symmetric.** Treats +1% gains and −1% losses similarly in their natural geometric sense.
- **Standard in quantitative finance.** Used in Yang et al. (2020) and FinRL.

#### 3.3.2 Why explicit transaction cost penalty?

The reward already reflects costs implicitly (executing a trade reduces $V_{t+1}$). However, **adding an explicit penalty term** on $|a_t - a_{t-1}|$:

- Provides a denser learning signal — the agent feels the cost of churn even on small moves.
- Discourages high-frequency oscillation (a common RL pathology where the policy learns to flip positions every step).
- Maps directly to my prior failure mode: the previous bot overtraded and fee erosion compounded losses.

#### 3.3.3 Reward shaping decisions deliberately NOT made

I considered and **deferred**:

- **Sharpe ratio as reward.** Mathematically clean but step-wise variance estimation is noisy; produces high-variance gradients. Yang et al. acknowledge this in their "future work" section. Defer to Phase 3 reward-shaping experiments.
- **Sortino / Calmar / drawdown penalty.** All produce non-Markovian reward (depend on history). Can be implemented but they violate the MDP assumption and need to be approximated.
- **Risk-aversion via utility function (e.g., $-\exp(-\alpha r)$).** Sound theoretical basis but adds a hyperparameter and changes the optimization landscape. Try if vanilla doesn't converge.

**Principle: vanilla log-return reward first. Only shape the reward if a specific failure mode demands it, and only one variable at a time.**

### 3.4 Transition Dynamics $P(s' | s, a)$

**Unknown.** This is the entire reason we use model-free RL.

We do not attempt to model the price process. The agent learns by interacting with a simulated environment built from historical data, and the transition function is implicitly the (deterministic, replay-based) historical price evolution combined with the (deterministic) portfolio accounting.

The implication: **our "transition dynamics" are historical data replay**, not a stochastic price model. This is a known weakness of backtest-based RL — every training rollout sees the same future. We mitigate via:

- Random episode start points (each episode samples a different 30-day window).
- Large dataset (>2 years of hourly data = ~17,500 hours).
- Walk-forward train/val/test split (§5).

### 3.5 Discount Factor $\gamma$

$$ \gamma = 0.95 $$

At 4-hour granularity, $\gamma = 0.95$ gives an effective planning horizon of approximately $1/(1-\gamma) = 20$ steps $\times$ 4h = 80 hours $\approx$ **3.3 days**. This matches the typical holding period for momentum/mean-reversion signals at this timeframe and avoids the agent attempting to plan over multi-week horizons where signal decays into noise.

Why not $\gamma = 0.99$ (the default I'd use at hourly granularity)? At 4H bars, $\gamma = 0.99$ gives a 100-step horizon = 400h ≈ 17 days. That's effectively asking the agent to plan two-week-ahead from price action, which the literature consistently shows is not predictable.

Alternative values to test in Phase 1 hyperparameter sweep: 0.90 (~10-step horizon, ~1.7 days), 0.97 (~33-step, ~5.5 days).

### 3.6 Episode Structure

- **Episode length:** 360 steps (60 days × 6 4H-bars/day). Longer calendar duration than the v1.0 plan to capture more regime variation per episode.
- **Episode start:** random index uniformly sampled from the training window (excluding the last 360 bars).
- **Initial state:** position = 0 (flat), portfolio value $V_0 = 1000$ (normalized — actual capital doesn't matter for policy learning).
- **Termination:** either 360 steps elapsed OR portfolio value drops below $0.5 V_0$ (50% drawdown circuit breaker, prevents training on terminal collapse trajectories where the agent learns from a doomed state).

Episode resets randomize the agent's exposure to different market regimes within the training window. At 4H resolution, 60 days exposes the agent to roughly one full short-term cycle.

---

## 4. Algorithm Selection: PPO

I will use **Proximal Policy Optimization** (Schulman et al. 2017), specifically the implementation in Stable Baselines 3.

### 4.1 Justification

| Criterion | PPO | DDPG | SAC | DQN |
|---|---|---|---|---|
| Continuous actions | ✅ | ✅ | ✅ | ❌ |
| Training stability | High | Low | Medium | Medium |
| Sample efficiency | Medium | High | High | Medium |
| Hyperparameter sensitivity | Low | High | Medium | Low |
| Reference implementations | Many | Some | Some | Many |
| Used in baseline trading papers | ✅ (Yang, FinRL) | ✅ (Yang) | Rare | Rare |

PPO's clipped surrogate objective explicitly prevents the kind of catastrophic policy updates that produced my previous model's collapse. The clipping factor $\epsilon$ (default 0.2) bounds the ratio between new and old policies, so a single bad batch cannot destroy the policy.

DDPG was a candidate but is widely known to be unstable without careful tuning, and the Yang et al. paper's quarterly Sharpe table shows it picked DDPG less than half the time despite training all three. Not worth the headache for v1.

SAC is appealing for sample efficiency but adds a temperature hyperparameter and is more complex to debug. Reserve as Phase 3 comparison.

### 4.2 Key hyperparameters (initial values)

| Hyperparameter | Value | Notes |
|---|---|---|
| `learning_rate` | 3e-4 | SB3 default, widely robust |
| `n_steps` | 2048 | Steps per rollout per env |
| `batch_size` | 64 | Minibatch for SGD updates |
| `n_epochs` | 10 | SGD passes per rollout |
| `gamma` | 0.95 | See §3.5 — calibrated for 4H bars |
| `gae_lambda` | 0.95 | GAE for advantage estimation |
| `clip_range` | 0.2 | Clipping epsilon — critical for stability |
| `ent_coef` | 0.01 | Entropy bonus — **higher than SB3 default of 0.0** to prevent the policy collapse that killed the previous model |
| `vf_coef` | 0.5 | Value function loss coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping |

The non-default `ent_coef = 0.01` is the most important deviation. **Entropy collapse is the diagnosis I assign to the previous model's failure**, and raising the entropy coefficient is the standard remedy.

---

## 5. Data Strategy

### 5.1 Source

- **Primary (training):** Kraken Futures REST API, BTC/USD perpetual (PI_XBTUSD or PF_XBTUSD), 4-hour OHLCV bars. Using Kraken throughout (training + paper trading) keeps the data distribution consistent end-to-end.
- **Backup:** CoinGecko (already integrated in existing bots) for sanity checks and continuity if Kraken API limits hit.
- **Funding rate data:** pulled separately from Kraken Futures API and merged onto the bar timestamps. See §6 for how it's incorporated.
- **Period:** 2022-01-01 to present.

### 5.2 Split

| Split | Period | Approx 4H bars | Use |
|---|---|---|---|
| Train | 2022-01-01 → 2024-06-30 | ~5,475 | PPO training |
| Validation | 2024-07-01 → 2025-06-30 | ~2,190 | Hyperparameter tuning, model selection |
| Test | 2025-07-01 → present | ~1,950 (held out) | Final eval only — touched once at end of project |

Note: 4H granularity gives fewer samples than the hourly v1.0 plan (≈17,500 → ≈5,475 training bars). This is intentional. The literature consistently shows lower signal-to-noise at sub-daily frequencies, and the reduced sample count is offset by the larger per-step learning signal.

**Critical rule: test set is locked. I will not run the model on test data until v1 is finalized.** Every time I peek, I introduce information leakage and degrade the honesty of the final numbers. This is the single most violated principle in applied ML.

### 5.3 Baselines to beat

A model is only meaningful relative to baselines. I will compare against:

1. **Buy-and-hold BTC.** The honest baseline. If I can't beat this risk-adjusted, the RL didn't add value.
2. **Constant fractional position (50% BTC, 50% cash).** Mean-variance-style passive.
3. **Naive momentum.** Long when 24h return > 0, flat otherwise.
4. **Random policy.** Sanity check that my policy is learning anything at all.

Reporting: cumulative return, annualized return, annualized volatility, Sharpe ratio, max drawdown, turnover. All metrics computed on validation and (once) test sets.

---

## 6. Constraints and Assumptions

| Assumption | Stated value | Realism |
|---|---|---|
| Transaction cost | 5 bps per side (0.05%) | Kraken Futures taker fee for retail; conservative |
| Slippage | 0 in backtest | Will be modeled in Phase 2 paper trading via Kraken testnet fills |
| Market impact | 0 | Realistic at $100-$10K capital on BTC perps |
| Liquidity | Unlimited at the 4H close price | Realistic for BTC perp on Kraken |
| Latency | 0 in backtest | Will surface in Phase 2 paper trading |
| Funding rate | Historical 8h funding applied at each crossing; applied as reward adjustment proportional to position size and time held | Realistic; perp-specific |
| Tax | Ignored | Out of scope |

Every assumption that gets relaxed in Phase 2 paper trading is a potential sim-to-real gap.

---

## 7. Success Criteria

**v1 is successful if and only if the following hold.** This is what I am optimizing for — not P&L.

### 7.1 Training-time criteria (must all hold)

1. **Training reward curve trends upward** over at least 1M timesteps before plateauing.
2. **Policy entropy stays above 0.3** through training (proxy: policy did not collapse).
3. **Value loss decreases** in tandem with policy improvement (critic is learning).
4. **No NaN/Inf in gradients** (numerical stability of the implementation).

### 7.2 Validation-time criteria (at least 3 of 4)

5. **Backtest Sharpe ratio > 0 on validation set.** Bar is low — just positive.
6. **Validation backtest does not pick a degenerate policy** (e.g., always-cash, always-long, oscillate-every-step).
7. **Turnover is below 1 round-trip per day on average** (sanity check that the cost penalty is working).
8. **Performance is at least competitive with the naive momentum baseline** on validation Sharpe.

### 7.3 Process criteria (must all hold)

9. **Every design decision in this document is reflected in the code** with comments referencing the doc.
10. **Training run is reproducible** (fixed seed, logged hyperparameters, versioned code).
11. **TensorBoard logs are preserved** for the final model and at least 2 failed runs.

**Profitability on the held-out test set is explicitly NOT a success criterion for v1.** It's a nice-to-have. Designing toward test-set profitability is how you get test-set overfitting.

---

## 8. Anticipated Failure Modes

Documenting these *in advance* so I have hypothesis space ready when something goes wrong.

| Failure mode | Symptom | Likely cause | First remedy |
|---|---|---|---|
| **Policy collapse** | Entropy → 0; policy always outputs same action | Insufficient entropy bonus; reward too sparse | Raise `ent_coef`; check reward scale |
| **All-cash policy** | Validation position is always 0 | Cost penalty too aggressive; agent learns "no trades = no losses" | Lower cost penalty; verify reward sign |
| **All-in policy** | Position is always ~1 | Bull market in training; agent learns "always long" | Add bear-market episodes to training; lower γ |
| **Step-step oscillation** | Position flips between 0 and 1 every hour | Reward noise > signal | Raise cost penalty; lengthen action smoothing |
| **Reward hacking** | Reward high but P&L bad | Bug in reward calculation; agent exploits sim quirk | Audit reward function against P&L tick-by-tick |
| **Training diverges** | Reward decreases over time | Learning rate too high; bad initialization | Lower LR; check observation normalization |
| **Sim-to-real gap** | Backtest great, paper trading bad | Slippage / latency / data freshness mismatch | Move to Phase 2 with careful instrumentation |

---

## 9. Postmortem Inputs from Previous Attempt

The previous model (`alpha_rl_aggressive_20250905_011903`) needs a written postmortem before v1 begins. **Questions to answer** (deferred to a separate `postmortem.md`):

1. What was the original state, action, and reward specification?
2. What was the training reward curve? Did it ever go up?
3. Was policy entropy logged? If so, when did it collapse?
4. What was the `ent_coef`?
5. Did the action distribution become degenerate (all mass at one value)?
6. When the `SmartTradingModel` wrapper was added, what did the underlying PPO actually do? Was it ignored entirely?

**The honest answer to several of these will likely be: "we don't know, because it wasn't logged."** That is itself the lesson — Phase 1 will log everything.

---

## 10. Deferred Decisions

Explicitly out of scope for v1. Each becomes a Phase 3+ experiment.

- **Multi-asset.** Portfolio-level RL with allocation across BTC, ETH, etc.
- **Higher frequency.** Move from 4H to 1H or 15m. Adds infrastructure burden and may not help SNR.
- **Leverage.** v1 caps notional at 1×. Adding leverage multiplies blow-up risk during exploration.
- **Funding-rate forecasting as a feature.** v1 incorporates realized funding into the reward but doesn't predict it; an extension would feed predicted funding into the state.
- **Deep state encoders.** LSTM/Transformer over raw OHLCV instead of hand-crafted features.
- **Ensemble methods.** Train multiple algorithms (PPO, A2C, DDPG), route by validation Sharpe — Yang et al. (2020) approach.
- **Risk-adjusted reward.** Sharpe / Sortino / utility-based.
- **Continuous online learning.** Retraining on rolling windows in production.
- **Sentiment / alternative data.** News, social, on-chain.

---

## 11. References

Primary:

1. Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
2. Yang, Liu, Zhong, Walid (2020). *Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy.* SSRN 3690996.
3. Liu, Yang, Gao, Wang (2021). *FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance.* SSRN 3955949.

Supporting:

4. Sutton & Barto (2018). *Reinforcement Learning: An Introduction.* 2nd ed.
5. Odermatt, Beqiraj, Osterrieder (2021). *Deep Reinforcement Learning for Finance and the Efficient Market Hypothesis.* SSRN 3865019.
6. Kritzman & Li (2010). *Skulls, Financial Turbulence, and Risk Management.* Financial Analysts Journal 66.

---

## Appendix A: Decision Log

| # | Decision | Chosen | Alternative considered | Rationale |
|---|---|---|---|---|
| 1 | Asset universe | BTC-USD only | Multi-asset (top 10 crypto) | Single asset reduces dimensionality and debugging cost |
| 2 | Time granularity | 4-hour bars | Hourly, daily, 15-min, 5-min | 4H trades off SNR vs sample count; matches typical retail crypto holding period |
| 3 | Action space | Continuous [-1, 1] | Discrete {sell, hold, buy}, continuous [0,1] | Continuous handles partial positions; signed range enables shorting |
| 4 | Direction | Long-short | Long-only | Kraken Futures testnet supports BTC perps; doubles policy expressive range |
| 5 | Execution venue | Kraken Futures testnet (paper) | Binance.US spot, Coinbase Advanced | Testnet is the only retail-accessible venue with realistic perp execution and no minimum |
| 6 | Algorithm | PPO | DDPG, SAC, DQN | Stability over sample efficiency; previous DDPG collapsed |
| 7 | Reward | Log-return − cost − funding | Sharpe, Sortino, utility | Vanilla first; funding term is perp-specific, not "shaping" |
| 8 | γ | 0.95 | 0.99, 0.97, 0.90 | Calibrated for 4H bars: ~3.3-day effective horizon |
| 9 | State features | Hand-crafted (~24 dims) | Deep encoder on OHLCV | Interpretability and debuggability over expressiveness |
| 10 | Episode length | 360 bars (60 days) | 180 bars (30d), full-history single episode | Captures regime variation; multiple per PPO rollout buffer |
| 11 | Train/val/test | 2.5y / 1y / held-out | k-fold, expanding window | Walk-forward respects causality |
| 12 | `ent_coef` | 0.01 (10x SB3 default) | 0.0, 0.001 | Explicit guard against the previous collapse |
| 13 | Success metric | Process criteria, not P&L | Test Sharpe, profitability | Avoids designing toward test-set overfitting |
| 14 | Environment implementation | Roll our own gym env | FinRL's StockTradingEnv | Self-built env is where most RL learning happens; FinRL hides the part that matters for understanding |

---

*End of Phase 0 document. Sign off here before proceeding to Phase 1.*
