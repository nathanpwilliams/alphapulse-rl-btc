# AlphaPulse

A crypto quant research platform for BTC and liquid alts, built to answer one
question: does a tradable edge survive realistic transaction costs? It is
structured like a miniature [Qlib](https://github.com/microsoft/qlib) — separate
packages for data loading, a Gymnasium RL trading environment, PPO training, an
HMM regime engine, a walk-forward backtester, and a live top-of-book quote
collector — and it was used to test six candidate edges on centralized venues
(Kraken Futures, OKX). All six were rejected on evidence, each with a mechanism
for *why*. The finding is a map of where retail-reachable edge is not; the method
— pre-registered success bars, controls that are allowed to fail, enforced
look-ahead guards, round-trip cost accounting — is the contribution.

## Results

Six candidates, each rejected against a bar fixed before the data was examined.

| # | Candidate | Measurement | Result | Mechanism |
|---|---|---|---|---|
| 1 | RL directional (PPO, BTC 4H) | Val reward on a chronologically later held-out slice | **500K-step run, 12-month window:** train reward +0.09 while val peaked +0.007 (~50–70K steps) then fell to −0.033. Smaller net ([32,32] vs [64,64]) raised explained_variance to 0.485 vs 0.442 but both converged to val −0.04…−0.047 by 500K | Overfits; data volume is binding. ~10 months of BTC 4H lacks enough distinct regimes. More data helped, more timesteps deepened overfitting, less capacity only softened it |
| 2 | Funding arb (single + cross-venue) | Funding collected per holding period vs round-trip cost, over 8,818 hourly Kraken rates (~368 days) | Single-venue: sign flips every **~10.2 h** (9.84% of intervals); break-even hold **~278 h** at 20 bps and still **~70 h** at an aggressive 5 bps. Cross-venue (92-day overlap): spread flips every **~6.1 h** — worse — with venue funding correlation only **+0.145** | Flip cadence is one to two orders of magnitude shorter than break-even hold, so a static position never earns its entry. The cross-venue spread needs correlated venues leaving a small stable residual; at +0.15, differencing compounds noise instead of cancelling it (mean \|spread\| 0.0705 bps/hr ≈ Kraken's own 0.0720) |
| 3 | Regime filter (HMM, long/flat) | Sharpe vs buy-and-hold, walk-forward labels, 10 bps/side | Sharpe **−0.392 vs +1.130**; annualized return −10.59% vs +51.01%; max drawdown −51.58% vs −53.55%; in market 32% of 8,165 bars | Halved vol (22.25% vs 45.73%) but surrendered the trend — much of BTC's return arrives in bars the rule flags unfavorable. Drawdown protection negligible despite 68% in cash |
| 4 | Regime-conditional RL | Train/val gap, baseline vs regime-augmented, identical otherwise | Gap widened **0.127 → 0.296**; train +0.0817 → +0.2126 while final val −0.0459 → −0.0837 | The 4-dim regime posterior added input capacity the data-limited policy spent on memorization. Val got *worse*, not suspiciously better — the signature of a clean causal feature |
| 5 | Cross-venue alt dislocation | Executable gaps (bid one venue vs ask the other, net of fees) from 11.2 days of live quotes | **BTC control failed**: 276 net-positive excursions, best +41 bps, 88.8% lasting ≤1.0 s, where the bar required ~0. Direction 87–97% one-sided across all 12 days. Capturable dollars (transient-direction, >1 s, **upper bound**), corrected: BTC $13, NEAR $11, UNI $1, ARB $0 per day | Two artifacts, no arbitrage: Kraken's ~1 s snapshot feed vs OKX's tick stream manufactures sub-second phantom gaps (stale-quote illusion, scales with volatility because it is latency); the persistent one-sided residue is the USD/USDT quote premium, which has no closing leg |
| 6 | Passive spread capture (market making) | Edge ratio = net half-spread ÷ mean adverse markout at τ=1 s | Every ratio below the bar of 1.0. ARB Kraken 0.57, NEAR Kraken 0.38, UNI Kraken −0.13; OKX negative throughout (−0.47 / −0.91 / −1.10). BTC control −3.2 / −2.5. No hour-of-day on any alt exceeds 0.45 | Two failure modes: on OKX professional makers quote alt spreads (2.0 bps) below a retail maker's 4 bps round-trip fee 100% of the time; on Kraken spreads are wide but 1 s adverse selection costs ~2.5× the net half-spread. Spread width and short-horizon adverse moves scale together |

All six rows are measured in this repo, by the modules in the map below, against
live venue data. Every figure is recorded in
`docs/journal/project_journal.md` as it was produced; `docs/findings.md` is the
narrative write-up.

## Method

The platform is built so a negative result is trustworthy. Five practices do the
work:

**Pre-registered bars.** Each candidate's success criterion was written down
before the data existed. Phase B's bar — BTC control excursions ~0, a >1 s
duration filter, directional symmetry — is what kept an eleven-day collection
from being rationalized into a marginal positive.

**Controls that are allowed to fail.** BTC and ETH ride along as
efficient-market controls. In Phase B the BTC control *did* fail, and that is the
result that overturned the whole analysis: without it, NEAR and UNI would have
read as a ~$50/day edge rather than a latency artifact.

**Enforced look-ahead guards.** Causality is asserted by tests, not by comment.
HMM regimes are refit walk-forward with train-window-only scaler statistics, each
bar labeled from its past alone, and the strategy acts on the prior bar:

- `tests/test_regime_detector.py:107` — `test_no_lookahead`
- `tests/test_regime_features.py:175` — `test_one_bar_lag`
- `tests/test_regime_backtest.py:137` — `test_lag_end_to_end`
- `tests/test_cross_venue.py:94` — `test_no_lookahead_uses_past_settlement_only`
- `tests/test_data.py:336` — `test_alignment_no_lookahead`
- `tests/test_spread_feasibility.py:88` — `test_kraken_resolution_floor_flag`
- `tests/test_executable.py:137` — `test_outage_flagged_not_treated_as_quiet`
- `tests/test_report.py:212` — `test_direction_split_flags_one_sided`

**Round-trip cost realism.** Costs are an explicit, sweepable parameter
(`--round-trip-bps`, `--cost-bps`) and every figure is reported net, never gross.
Both legs, entry and exit: 10 bps/side in the regime backtest, 2 bps/side maker
fees (4 bps round trip) in the spread-capture analysis, a 20 bps hurdle in the
Phase A screen.

**Single-variable experiments.** The three overfitting experiments each changed
exactly one thing — data window, then timesteps, then network architecture —
holding seed, envs and hyperparameters fixed, so each outcome is attributable.
The same discipline makes the Level 3 baseline byte-identical to the augmented
run except for the regime feature.

**Published corrections.** OKX quotes size in contracts, Kraken in base units;
`min(bid_size, ask_size)` across them mixed units. The fix (ctVal: BTC 0.01,
NEAR 10, UNI 1, ARB 10) is recorded in the journal alongside the superseded
figures, with UNI unchanged to the cent because its ctVal is 1 — the expected
check that the fix touched only non-unit contracts.

## Module map

| Package | Contents |
|---|---|
| `src/data/` | `load_kraken.py` (Kraken Futures OHLCV + funding), `load_okx.py` (OKX candles + funding), `load_cross_venue.py` (aligned multi-asset basket), `collect_top_of_book.py` (live websocket quote collector) |
| `src/env/` | `trading_env.py` — Gymnasium trading environment, optional regime-posterior state |
| `src/training/` | `train.py` — PPO scaffold with chronological validation callback and best-by-val checkpointing |
| `src/strategies/funding/` | `analyzer.py` (single-venue funding arb), `cross_venue.py` (Kraken–OKX funding spread, cadence-normalized) |
| `src/strategies/regime/` | `detector.py` (HMM with AIC/BIC state selection), `features.py` (causal walk-forward posteriors), `backtest.py` (walk-forward regime filter vs buy-and-hold) |
| `src/strategies/dislocation/` | `scanner.py` (Phase A candle screen), `executable.py` (Phase B quote-level executable gaps), `streaming.py` (chunked driver), `report.py` |
| `src/strategies/mm/` | `spread_feasibility.py` — passive spread-capture economics |

**327 tests**, all passing, mirroring the source tree under `tests/`. They are
synthetic-data unit tests that do not hit the network; live-data correctness is
left to each module's CLI smoke test.

```
pytest tests/ -v
```

## Quick start

Python 3.9+ (developed on 3.9.6; every module carries
`from __future__ import annotations` for 3.9 compatibility — 3.11+ is
recommended if you are starting fresh).

```bash
git clone https://github.com/nathanpwilliams/alphapulse-rl-btc.git
cd alphapulse-rl-btc
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # data loaders, env, collector, analyzers
```

`requirements-train.txt` adds the heavy dependencies — PyTorch,
Stable-Baselines3, TensorBoard, `hmmlearn` — needed for RL training, the regime
modules, and the **full** test suite:

```bash
pip install -r requirements.txt -r requirements-train.txt
pytest tests/
```

PyTorch is ~2 GB. Without it the funding and dislocation analyzers still run,
but `tests/test_training.py` cannot import and the regime modules cannot fit an
HMM.

## Reproducing

The three fastest reproductions need no stored data — they pull from public
endpoints and print their analysis:

```bash
python -m src.strategies.funding.analyzer        # single-venue funding arb (row 2)
python -m src.strategies.funding.cross_venue     # Kraken-OKX funding spread (row 2)
python -m src.strategies.regime.detector         # HMM regime structure (row 3 input)
```

Costs are sweepable: `--round-trip-bps 10`, `--cost-bps 20`.

Longer runs: `python -m src.strategies.regime.backtest` is the walk-forward
regime filter (~4.3 min on full history — the per-bar causal decode is the
expensive-but-honest part). `python -m src.strategies.dislocation.scanner`
fetches 30 days of 5m candles from both venues. The Phase B and spread-capture
analyses (`src.strategies.dislocation.executable`,
`src.strategies.mm.spread_feasibility`) read the 4.4 GB quote set, which is not
in this repo; `docs/tob_collector_runbook.md` documents how it was collected.

## Infrastructure

Phase B's quote set was collected by a websocket collector running as a systemd
service under an unprivileged user on a DigitalOcean droplet: **11.2 days, 71.5M
rows, 4.4 GB, 268.2 clean hours**, zero service restarts, one 397 s Kraken outage
masked rather than interpolated. Analysis runs on a Mac against a chunked
streaming driver (peak RSS 0.78 GB) wrapping the unmodified analyzer.
Deployment, health checks and data pull are in
`docs/tob_collector_runbook.md`.

## Limitations

These bound what the results above can claim:

- **Funding history is ~1 year.** Kraken's funding record gave 8,838 hourly
  rates (2025-07-13 → 2026-07-16) — a rolling ~1-year window, so the count varies
  with fetch date. The ~10 h sign-flip cadence is measured over one year of one
  regime mix.
- **Kraken's quote feed has a ~1 s floor.** It publishes snapshots, not ticks:
  43 adverse events/hour against OKX's 1,123/hour for the same asset at the same
  realized volatility. Every Kraken edge ratio is therefore an *upper bound*, and
  sub-second cross-venue excursions against Kraken are not trustworthy.
- **Quotes, not trades.** Fill rates are not observable from top-of-book data, so
  the spread-capture analysis gives per-fill economics only — not a P&L.
- **USD/USDT basis.** Kraken Futures perps are USD-quoted, OKX USDT-quoted, so
  same-quote pairing was infeasible across the 173 shared assets. The basis rides
  as a structural premium (~2.4 bps constant on BTC, rising to ~9.4 bps on ARB)
  that the analyzers separate from transient dislocation rather than eliminate.
- **RL data volume.** The directional result is one asset at one resolution:
  7,123 train / 2,375 val bars of BTC 4H. "Overfits" is a statement about this
  data volume, not about RL for trading in general.

## What is untested

On-chain and DEX venues are the obvious next arena — different microstructure,
different participants, and the latency asymmetry that killed row 5 does not
apply the same way. That is untested here, and stating it is not a plan.

## License

MIT — see [LICENSE](LICENSE).
