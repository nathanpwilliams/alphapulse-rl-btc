# AlphaPulse RL Trading Bot — Project Journal

A running record of decisions, bugs, debugging stories, and lessons across the
project. Each session adds an entry at the bottom. Most recent entries last.

This is a technical project record: a running log of decisions, experiments,
findings, and lessons across the project's development.

---
## 2026-06-22 — Phase 2: First PPO run + overfitting finding

**Status:** Phase 2 underway. Training pipeline complete and smoke-tested.
First PPO run executed end-to-end. 62 tests passing.

### What I did

Built and committed the PPO training scaffold (`src/training/train.py` ~400
lines, `tests/test_training.py`, 13 tests). Ran the first end-to-end training:
57,344 timesteps across 4 parallel envs, completed in under a minute on local
Apple Silicon (Kraken data was cached, so no network round-trip). Inspected the
run in TensorBoard.

### What worked

The full pipeline wired together on the first real run with no integration
bugs — data → chronological split → vectorized env → PPO → callbacks → save.
Training reward (`rollout/ep_rew_mean`) rose from ~-0.19 to ~-0.08, showing the
agent learning to reduce losses (mostly by trading less and paying fewer
transaction costs). Entropy decreased gently (entropy_loss -1.435 → -1.39),
confirming the policy is NOT collapsing — the ent_coef=0.01 is doing its job.

### What didn't / what surprised me

Validation reward (`eval/mean_reward`) DECLINED over the run, from ~0 to ~-0.03,
while training reward rose. This train-up/val-down divergence is the textbook
overfitting signature, and it appeared almost immediately (~step 10K). With only
~7 months of training data after the val split, the agent memorizes the training
slice fast. The value function (`explained_variance`) improved from ~-13 to ~-1.2
but is still negative — the critic hasn't learned to value states well at this
step count.

### What I learned

- The validation harness earned its keep on the very first run: it caught
  overfitting that training metrics alone would have hidden.
- "Train longer" is the wrong next move — it would deepen overfitting. The real
  levers are more data, early stopping at peak val reward, smaller model
  capacity, and stronger regularization.
- best_model.zip (early-stopped at peak val) and final.zip (overfit at 57K) are
  genuinely different models. Saving only the final model — the default behavior —
  would have shipped the overfit one.

### What's next

Address overfitting before any long run: widen the training window toward
Kraken's full ~1-year funding history, rely on the best-by-val checkpoint, and
experiment with smaller network capacity ([32,32]) and higher ent_coef. Treat
each change as an isolated experiment and compare val curves across runs.

---

## 2026-07-14 — Infrastructure: private GitHub repo, SSH auth, authorship cleanup

**Status:** Project now version-controlled on GitHub (private). All infrastructure
debt cleared. Ready to resume ML work on the overfitting problem.

### What I did

Pushed AlphaPulse to a private GitHub repo under nathanpwilliams, using SSH
(key-based) auth rather than HTTPS-with-token. Generated a dedicated Ed25519 key,
registered it with GitHub, configured `~/.ssh/config`. Rewrote all 8 commits'
authorship from a stale identity (sunstargroup) to Nathan Williams with a
privacy-preserving GitHub noreply email, then set that as the global git default.
Fixed the `code` shell command (was broken by macOS App Translocation — VS Code
had to run from /Applications for the symlink to point at a stable path).

### What worked

SSH auth succeeded and confirmed the GitHub account is healthy — the earlier
"account locked" worry was actually just an unconfigured SSH setup, never an
account problem. The commit-authorship rewrite (git filter-branch) cleanly
changed all 8 commits and correctly carried the phase1-complete tag to the new
hash. Repo pushed with full history intact and verified private.

### What didn't / what surprised me

Several small snags: typed sudo password into an SSH passphrase prompt by mistake
(harmless — nothing stored or sent); the `code` command failed repeatedly with
EACCES until root-causing it to App Translocation (VS Code running from a
temporary randomized path, so its symlink pointed at a location that kept
disappearing). Fixed by running VS Code from /Applications and clearing the old
root-owned symlink with sudo before reinstalling.

### What I learned

- git commit authorship is just metadata, not an auth link — GitHub authenticates
  by SSH key, not by the author field. But mismatched authorship means commits
  won't link to your profile/contribution graph unless the email is verified on
  the account.
- The reversible choice is private-first: you can always make a private repo
  public later, but you can never truly un-publish a public one.
- Published alpha is dead alpha — a genuinely profitable strategy loses its edge
  once public, so the profitable version (if it ever exists) stays private and any
  public portfolio piece is a deliberately sanitized/educational variant.
- `git branch backup-before-rewrite` does NOT survive a filter-branch that targets
  --branches (it gets rewritten too); a real pre-rewrite backup needs a `.git`
  copy or `git bundle`.

### What's next

Resume the ML work: tackle the overfitting finding from the first PPO run
(more data, early stopping, smaller network, stronger regularization).

---
## 2026-07-16 — Phase 2 experiment: more data reduces overfitting

**Status:** First controlled overfitting experiment complete. Widening the
training window from ~9mo to ~12mo visibly reduced the train/val divergence.

### What I did

Ran a single-variable experiment against the first PPO run's baseline: kept
everything identical (50K timesteps, 4 envs, seed 42, all hyperparameters) and
changed ONLY the data window, from 9 months (2025-08 -> 2026-05) to the full
~12 months available from Kraken's funding history (2025-07-15 -> 2026-07-14).
Verified the real funding range first via fetch_funding_rates (8,838 hourly rates,
2025-07-13 -> 2026-07-16).

### What worked

Overfitting visibly decreased. Baseline: validation reward declined into negative
(~-0.03) from step ~10K while training reward rose — classic memorization. This
run: validation reward stayed small-but-POSITIVE (~+0.005) with no sustained
downward trend, while training reward rose similarly (~-0.17 -> -0.14). The
train/val divergence that defined the baseline was largely absent. value_loss
declined cleanly (0.0017 -> 0.0003); explained_variance improved to ~-1.07.

### What didn't / what surprised me

Because the two runs train/validate on different market periods, absolute reward
values aren't directly comparable — only the train-vs-val SHAPE is. The val
reward's positive band is on a very zoomed-in axis (total range ~0.0003), so this
is "not degrading on held-out data," not "now profitable." Explained_variance is
still negative — the critic needs more training to value states well.

### What I learned

- Data was a genuine bottleneck: ~3 extra months of training data measurably
  reduced overfitting at identical everything-else. This validates fixing data
  before scaling timesteps.
- The correct cross-run comparison for different data windows is the train/val
  divergence pattern, not raw reward values.
- Single-variable discipline paid off: because only the data changed, the reduced
  overfitting is cleanly attributable to data, not confounded with other changes.

### What's next

Scale timesteps on the 12-month window (500K, 8 envs) and check whether val reward
keeps holding while explained_variance finally crosses into positive. Then consider
smaller network capacity and ent_coef tuning as further overfitting controls.

---
## 2026-07-16 — Phase 2 experiment: timesteps are double-edged

**Status:** Second controlled experiment. Scaled timesteps 57K -> 500K on the
12-month dataset. Fixed the critic but deepened overfitting — a key lesson in
how training duration behaves.

### What I did

Held the 12-month data window and all hyperparameters fixed; changed timesteps
(50K -> 500K) and n_envs (4 -> 8). Compared directly against the more_data_12mo
run in TensorBoard. Training took ~2.3 min.

### What worked

The value function finally learned: train/explained_variance crossed from ~-12
into POSITIVE territory, ending ~+0.39 (the 57K run never left negative). The
critic now explains ~39% of return variance. Training reward climbed all the way
to positive (~+0.09). value_loss declined cleanly and stayed stable (~0.0004).

### What didn't / what surprised me

Overfitting got WORSE with more timesteps. eval/mean_reward peaked early (~step
50-70K, ~+0.007) then declined steadily to negative (~-0.033 around step 130K),
while training reward kept rising to +0.09. The train-up/val-down divergence — the
overfitting signature — was more pronounced at 500K than at 57K. Training longer
let the agent memorize the ~10-month training period more thoroughly, at the cost
of generalization.

### What I learned

- Timesteps is a double-edged variable: it fixed the critic (needed more steps to
  learn value prediction) but deepened policy overfitting (extra steps fit noise
  once real signal was exhausted).
- The validation curve reveals BOTH the overfitting AND the optimal stopping point:
  val reward peaked ~50-70K, so that's the model worth keeping. best_model.zip
  captured it; final.zip at 500K is the overfit endpoint.
- Data and timesteps interact: more data pushed the val peak later (7mo overfit at
  ~10K steps; 12mo at ~50-70K). More useful timesteps require more data, not just
  more steps.
- The training reward curve is a trap — +0.09 looked like profitable learning but
  was memorization. Only the val curve told the truth.

### What's next

Fight overfitting directly with single-variable experiments: smaller network
([32,32] vs [64,64]) to reduce memorization capacity, and/or higher ent_coef
(0.02-0.05) for more regularization. Verify which step best_model.zip was saved at.
Hold data and timesteps fixed so any change is attributable to the regularization
lever.

---
## 2026-07-17 — Phase 2 experiment: smaller network + the data-constraint conclusion

**Status:** Third single-variable experiment complete. Reduced capacity
([64,64]→[32,32]) softened mid-training overfitting and kept the critic healthy,
but both nets converged to similar poor val reward by 500K. Cross-experiment
conclusion: data volume is the binding constraint.

### What I did

Added a --net-arch CLI flag (captured in run_config.json; a stale test fixture
caught the schema change and was fixed). Ran smaller_net_32: identical to
more_data_500k except net_arch [32,32] vs [64,64].

### What worked

Reduced capacity softened overfitting through the middle of training (100K-300K):
the smaller net's val reward held higher than the larger net's overfitting trough.
Critically, this cost nothing on the critic — explained_variance ended slightly
HIGHER for the smaller net (0.485 vs 0.442). So: less overfitting, equal/better
value function, fewer parameters, faster. The 24-dim state simply doesn't need a
large network; the extra capacity in [64,64] mostly enabled policy memorization.

### What didn't / what surprised me

By 500K, both nets CONVERGED to similar poor final val reward (~-0.04 to -0.047).
The smaller net was better *along the way* but not *at the endpoint*. Capacity
reduction delays and softens overfitting but doesn't escape it — because the real
constraint isn't architecture.

### What I learned

- Network capacity is a real but limited overfitting lever: it improves the
  trajectory and the best-checkpoint window, not the 500K endpoint.
- Smaller network here is strictly better engineering: equal generalization,
  better critic, fewer params, faster — no downside found.
- CROSS-EXPERIMENT CONCLUSION: three experiments (more data helped; more timesteps
  deepened overfitting; smaller net only softened it) all point to the same
  binding constraint — data volume. ~10 months of BTC 4H history lacks enough
  distinct regimes for a policy to generalize to a held-out period. The answer
  isn't more tuning; it's more/more-diverse data.

### What's next

Decision fork: (1) accept data volume as the documented constraint and use the
early-stopped smaller-net best_model.zip as the best-effort model, or (2) attack
the constraint directly with multi-year data from a paid provider (CoinAPI/
Amberdata) or additional assets. Leaning toward documenting the finding and, if
continuing, sourcing more data rather than further tuning.

---
## 2026-08-22 — Level 2: regime-as-filter backtest (tradeability test)

**Status:** Built the regime-filtered backtest to test whether Level 1's validated
regime structure translates into a tradeable edge — regime filter (long favorable
regimes / cash otherwise) vs buy-and-hold, on walk-forward regime labels with honest
cost and detection-lag accounting.

### What I did

Built src/strategies/regime/backtest.py, reusing the Level 1 HMM detector. Regimes
assigned walk-forward (HMM refit on a rolling cadence, each bar labeled only from its
past) to avoid look-ahead; strategy acts on the prior-bar regime to model detection
lag. Long/flat filter: hold the asset in favorable regimes (positive drift, below-
median vol as known at decision time), cash otherwise, with per-side transaction
costs. Compares Sharpe, max drawdown, annualized return, time-in-market, and turnover
against buy-and-hold. Parameterized for asset/resolution with per-asset AIC/BIC state
selection (no hardcoded regime count) and detector/eval separation, as seams for the
future multi-asset regime scanner. 20 tests, look-ahead and lag guards non-negotiable.

To support walk-forward without look-ahead, the detector was refactored minimally: the
z-score scaler was split into `_scaler_stats` + `_apply_scaler` (so test bars are
standardized with training-window statistics only), and a reusable `fit_model`
(fit-with-scaler, no decode) was extracted; `fit_regimes` now builds on it and the
Level 1 single-fit diagnostic path is unchanged.

### What the backtest showed

Live PF_XBTUSD 4H, 2022-11-30 → 2026-08-21, 8,165 bars traded. n_states auto-selected
= 4 (BIC, from first 1,500 bars), 46 walk-forward refits, 10 bps/side cost. The filter
was in market 32% of the time with 592 position changes.

| Metric | Regime filter | Buy & hold |
| --- | --- | --- |
| Annualized return | -10.59% | +51.01% |
| Annualized vol | 22.25% | 45.73% |
| Sharpe (rf=0) | -0.392 | +1.130 |
| Max drawdown | -51.58% | -53.55% |
| Calmar | -0.205 | +0.952 |

Sharpe uplift (filter − hold) = -1.522; drawdown change = +1.97 pp (filter marginally
shallower). The filter roughly halved volatility but surrendered essentially all
return and posted a negative Sharpe; drawdown protection was negligible despite being
in cash 68% of the time. Numbers only — no viability verdict.

### What I learned

- Walk-forward labels are far more conservative than Level 1's in-sample Viterbi
  decode: the regime known only from the past is lagged and noisier than the
  full-sample path. This is exactly why in-sample-labelled backtests are worthless —
  the honest label is what determines tradeability, and it is a weaker signal.
- The "positive drift + below-median vol" favorable rule kept the book in cash 68% of
  the time, and much of BTC's return over this window came in bars the rule flags as
  unfavorable (higher-vol up-moves). Filtering those out surrendered the trend — the
  filter cut vol ~in half (22% vs 46%) but gave up the upside with it.
- Drawdown barely improved (−51.6% vs −53.6%) despite low exposure: being in cash most
  of the time did NOT reliably sidestep the large declines. With one-bar detection lag
  and a 180-bar refit cadence, the filter re-enters into/holds through drawdowns rather
  than avoiding them.
- Turnover is modest: 592 changes over 8,165 bars (~7% of bars flip), so transaction
  cost is not the primary driver of underperformance — the long/flat signal itself is.
- Compute: honest walk-forward (per-bar causal decode from each refit boundary + 46
  refits × 5 restarts) ran ~4.3 min on full history. The per-bar decode is the
  expensive-but-honest part; a future scanner would cache fits and label incrementally.

### What's next

Interpretation of tradeability happens with the lead. If regime-filtering improves
risk-adjusted return, Level 3 (regime-conditional RL / regime as a state feature —
the direct fix for the earlier single-policy overfitting) is justified, and the
multi-asset recurring scanner becomes the infrastructure build. If not, regimes are
real-but-not-exploitable-this-way and we reconsider the filter design.

---
## 2026-09-10 — Level 3: regime as a causal RL state feature (baseline vs augmented)

**Status:** Tested whether feeding the causal regime posterior to the PPO agent as a
state feature improves generalization vs a no-regime baseline — the intended
application of the regime work and the direct fix for the original single-policy
overfitting. Both runs identical except the regime feature.

### What I did

Added include_regime to EnvConfig (off by default; baseline behavior byte-identical).
Built src/strategies/regime/features.py: causal walk-forward HMM posteriors (soft
state probabilities) per bar, reusing Level 2's walk-forward machinery (train-window
scaler, refit cadence, one-bar lag) so no future info enters the state. Refactored the
Level 2 walk-forward into a shared causal skeleton (`_walk_forward_per_bar`) so the
hard-label and new soft-posterior paths cannot diverge; the soft path takes
`predict_proba(window)[-1]`, which at a sequence's final position is the FILTERED
(causal) posterior. Wired --include-regime into training. Ran L3_baseline (no regime)
and L3_regime (regime posterior appended) — identical timesteps/envs/seed/net-arch/data
(500K, 8 envs, seed 42, [32,32], 2022-04-01→2026-08-01, 7,123 train / 2,375 val bars),
only the feature differs. 17 new tests; look-ahead and lag guards non-negotiable.

n_states auto-selected = 4 (BIC, walk-forward on the first 1,500 bars) — agreeing with
Level 1's full-sample finding. ~7,986 / 9,498 bars carried a genuine lagged causal
posterior; the rest (feature warmup + pre-first-fit window + the first labeled bar) got
a neutral uniform posterior.

### What the comparison showed

| Metric | Baseline (no regime) | Regime-augmented |
| --- | --- | --- |
| Final train reward (rollout/ep_rew_mean) | +0.0817 | +0.2126 |
| Final val reward (eval/mean_reward) | -0.0459 | -0.0837 |
| Peak val reward | +0.0097 (410K) | +0.0078 (450K) |
| Final explained_variance | +0.080 | +0.045 |
| Train/val gap (final) | 0.127 | 0.296 |

The regime feature INCREASED the train/val gap (~0.13 → ~0.30): training reward jumped
markedly (+0.08 → +0.21) while final validation reward got worse (-0.046 → -0.084) and
peak validation was marginally lower. Numbers only, no verdict.

### What I learned

- Adding the 4-dim regime posterior gave the policy extra input capacity, which on this
  data-limited problem it spent memorizing the training period rather than
  generalizing: train reward nearly tripled while val degraded. This is the
  "more input dims = more overfit capacity on limited data" outcome flagged as a real
  possibility — consistent with the standing finding that data volume, not features or
  architecture, is the binding constraint.
- Crucially, the augmented agent looked WORSE on validation, not suspiciously better —
  exactly the signature expected when there is NO look-ahead leak. A regime feature that
  leaked the future would have inflated validation reward; instead the honest causal
  walk-forward + one-bar lag (guarded by the non-negotiable no-look-ahead/lag tests)
  produced a feature that simply didn't help. That the result is unimpressive is itself
  evidence the causality is clean.
- The soft posterior's causality rests on a clean property: the forward-backward
  marginal at a sequence's last position equals the filtered posterior (beta_T = 1), so
  `predict_proba(window)[-1]` uses no data beyond the window end. This let the soft path
  reuse the exact refit/scaler skeleton of the hard path.
- Cost: the causal posterior precompute (walk-forward, 45 refits, per-bar predict_proba)
  ran ~4.4 min on the full history — the same order as Level 2. Each 500K PPO run was
  ~2.3 min; the regime run's only added cost is the one-time precompute.
- Env plumbing stayed clean: with include_regime=False the observation space and all
  existing env tests are byte-identical; the regime columns are appended last and read
  straight from the pre-lagged array, so the env cannot introduce look-ahead.

### What's next

Interpretation with the lead. Per prior agreement this is the last regime experiment
before a strategic reassessment: if the regime feature improves generalization, the
regime work becomes central (regime-conditional RL + the multi-asset scanner); if not,
BTC 4H regime structure — though real — doesn't convert to edge with these tools, and
the arena itself gets reconsidered.

---
## 2026-09-24 — Pivot: cross-venue price dislocation scanner (Phase A, multi-asset)

**Status:** After four BTC-4H strategies (RL directional, funding arb, regime filter,
regime-conditional RL) failed to beat buy-and-hold — common cause: efficient,
data-limited arena plus predictive edges — pivoted to a STRUCTURAL edge in a LESS-
EFFICIENT arena: cross-venue (Kraken Futures vs OKX) price dislocation across a basket
of alts at 5m, with BTC/ETH as efficient-market controls. Built the Phase A screen.

### What I did

Probed both venues first: 173 shared base assets; Kraken Futures perps are USD-quoted,
OKX USDT-quoted, so same-quote pairing is infeasible — compared USD vs USDT and let the
basis ride as a structural premium the analyzer separates from transient dislocation.
Basket: BTC, ETH (controls) + SOL, XRP, DOGE, LTC, NEAR, UNI, SUI, ARB (~$1.2B → ~$0.1B
24h notional). 5m depth: 30d used (Kraken 35d gap-free via the paginating loader; OKX
≥31d). Added fetch_okx_ohlcv, load_cross_venue_basket, and
src/strategies/dislocation/scanner.py: tight-tolerance alignment with no forward-fill,
gap stats, structural premium (signed mean + sign stability), excursions above cost on
raw AND mean-removed gap, AR(1) half-life, cross-asset table. 18 tests; 196 total.

### What the scan showed (30d, 5m, 20 bps hurdle)

Controls clean: BTC/ETH ~0.1 transient excursions/day, mean |gap| ~3 bps, half-life
18–34 min. Alts show a monotonic liquidity gradient: UNI 7.4, NEAR 17.9, ARB 41.9
transient excursions/day; ARB mean |gap| 15.5 bps, p95 38.5 bps, half-life 1.6 min.
Structural premium positive on every asset and rising with illiquidity (BTC +2.4 →
ARB +9.4 bps).

### What I learned

- The screen behaves as theory predicts: no exploitable gap where the market is
  efficient, rising dislocation as liquidity thins. First positive-direction
  confirmation in the project.
- Three reasons the candle result cannot yet be believed: (1) ARB's 1.6-min half-life
  is shorter than the 5m sampling interval, so most of an excursion reverts between
  observations; (2) candle closes are last-trade prints, which go stale on thin books
  and create phantom gaps; (3) the 20 bps hurdle assumes a spread the candles can't see,
  and thin alt perps may have 10–20 bps spreads each side. All three are resolved only by
  bid/ask quotes.
- The USD/USDT quote mismatch shows up as a ~2.4 bps constant on BTC; the larger alt
  premia are venue friction/staleness, correctly excluded from transient counts.

### What's next

Phase B: live top-of-book collection (public websockets, both venues) for ARB, NEAR,
UNI + BTC control over ~5–7 days, then recompute EXECUTABLE gaps (bid on one venue vs
ask on the other, net of fees) and rerun the excursion analysis. Candidates are
believed only if executable excursions survive.

---
## 2026-10-13 — Phase B: live top-of-book collection and executable-gap analysis — cross-venue alt arb rejected

**Status:** Collected 11.2 days of live best bid/ask from Kraken Futures and OKX
(ARB, NEAR, UNI; BTC control) on a dedicated droplet: 71.5M rows, 4.4 GB, zero
service restarts, one 397 s Kraken outage masked. Ran the executable-gap analysis
against a success bar fixed before any data existed. Every criterion failed. Cross-
venue Kraken–OKX arbitrage on these alts is rejected.

### What I did
Deployed the websocket collector as a systemd service under an unprivileged user
(Phase 0–2, each gated on approval); health check at day 11; stopped, rsynced 71.5M
rows to the Mac, and added a chunked streaming driver (peak RSS 0.78 GB) around the
unmodified analyzer. Reported excursion counts, durations, size-capped capacity,
daily capturable dollars, hour-of-day and weekday/weekend splits, and direction.

### What the data showed
- BTC control: 276 net-positive excursions, best +41 bps, 88.8% lasting ≤1.0 s —
  not ~0 as required. BTC dollars concentrate on two volatile days.
- Alts: ARB 272, NEAR 527, UNI 508 excursions/day; 50–65% ≤1.0 s. Raw mean
  capturable: ARB $3/day, NEAR $54, UNI $44 (medians $3/$28/$22).
- Restricting to excursions >1 s (above the Kraken ticker floor): ARB $2, NEAR $25,
  UNI $17 per day.
- Direction: 87–97% of alt excursions are sell-Kraken/buy-OKX, persistent all 12
  days, matching Phase A's structural premium ordering. The mean-reverting minority
  direction totals $2 / $61 / $135 over 268 h (ARB/UNI/NEAR).

### What I learned
- The control failing is the key result: Kraken's ~1 s snapshot feed vs OKX's tick-
  by-tick stream manufactures sub-second phantom gaps during fast moves (stale-quote
  illusion). Apparent dollars scale with volatility because they are latency, not
  mispricing. Without the control, NEAR/UNI would have read as a ~$50/day edge.
- The persistent one-sided component is the USD/USDT quote-currency premium leaking
  through raw (non-mean-removed) gaps. It has no closing leg without paying the
  transfer friction that sustains it, so it is not an arbitrage.
- Phase A's liquidity ordering (ARB most dislocated) did not reproduce at quote
  resolution; ARB has the smallest capacity and least money.
- Pre-registering the bar (BTC ~0; >1 s filter; symmetry) is what kept a ten-day
  wait from turning into a rationalized marginal result.
- Ops: SIGTERM did not run the collector's final flush (no handler installed;
  ~200 rows lost); macOS openrsync rejects --info=stats2 and the first pull was a
  silent no-op caught by a row-count check; collector RSS grew to 429 MB over 11
  days (bounded run, but a leak signature to check before any month-long run).

### Where the project stands
Five strategies rejected on evidence with mechanisms: RL directional (overfits),
funding arb (sign-flip cadence), regime filter (lag), regime-conditional RL
(dimensional overfit), cross-venue alt dislocation (structural premium + latency
artifact; transient edge ≈ single-digit $/day). Common thread: edges reachable from
retail infrastructure on centralized venues are already captured by faster capital;
the recorded sub-second excursions are other participants' trades. The thin-alt
bid-ask spread that made the arb uncapturable is the one income source on the
original list not yet measured (passive spread capture / market making).

### What's next
Decision with the lead: consolidate the platform as the deliverable (README/writeup
of the five-rejection arc), optionally one data-only analysis of passive spread-
capture economics on NEAR/UNI using the existing 11-day quote set.

---
## 2026-10-14 — Path B: passive spread-capture feasibility — market making rejected

**Status:** Data-only analysis of market-making economics on NEAR/UNI (ARB extra,
BTC control) from the 11-day top-of-book set. Pre-registered bar: edge ratio
(net half-spread ÷ mean adverse markout at τ=1 s) < 1 on both alts → dead. Result:
dead on both, on both venues, in every hour.

### What I did
Built src/strategies/mm/spread_feasibility.py (streamed, 1 s grid, outage masked):
spread distributions, net half-spread after base-tier maker fees (2 bps/side both
venues), realized σ at 1/5/30 s, a naive touch-quoter adverse-event model
(trade-through = certain adverse fill) with markouts, edge ratios, touch size,
300 s inventory drift, hour-of-day and weekday/weekend splits. Fill rates are not
observable from quotes; the analysis gives per-fill economics only.

### What the data showed (τ=1 s)
- BTC control: edge ratio −3.2 / −2.5 — correctly unmakeable at 1 s.
- NEAR: Kraken 0.38 (median spread 8.8 bps, net half +2.4, adverse cost 6.4);
  OKX −0.47 (spread 2.0 bps — below the 4 bps round-trip maker fee 100% of the time).
- UNI: Kraken −0.13; OKX −0.91. ARB: Kraken 0.57; OKX −1.10.
- No hour-of-day on any alt exceeds 0.45. Weekends cut adverse rates 30–60% but
  leave ratios unchanged.
- Kraken's ~1 s snapshot undercounts adverse events (43/hr vs 1,123/hr on OKX for
  the same asset and σ), so Kraken ratios are upper bounds.

### What I learned
- Two failure modes: on OKX, professional makers already quote alt spreads below a
  retail maker's fees and inside one second of volatility — a latency/fee-tier arms
  race; on Kraken, spreads are wide but adverse selection at 1 s costs ~2.5× the net
  half-spread. Spread width and short-horizon adverse moves scale together, which is
  why no venue, hour, or weekday/weekend split clears 1.
- OKX quotes sizes in contracts (ctVal); mixing units with Kraken base sizes was a
  latent bug that also affected Phase B capacity figures (corrected separately).

### Where the project stands
Six strategies tested on the original list; all rejected on evidence with a
mechanism. The platform and methodology are the deliverable.

---
## 2026-10-14 — Correction: Phase B capacity/dollar figures (OKX contract units)

Phase B's executable analysis computed min(bid_size, ask_size) with Kraken sizes in
base units and OKX sizes in contracts. Corrected with ctVal (BTC 0.01, NEAR 10,
UNI 1, ARB 10). Corrected figures (11.2 days, 268.2 clean hours; excursion counts,
bps, durations and direction splits are unchanged — this is purely a size-unit fix):

| Asset | Capacity median/p90 $ | $/day all | $/day >1 s | $/day transient >1 s | Total raw $ | OKX leg bound |
| ----- | --------------------: | --------: | ---------: | -------------------: | ----------: | ------------: |
| BTC   |            76 / 26,509 |        50 |         16 |                   13 |         559 |         12.0% |
| ARB   |              245 / 911 |        20 |         11 |                    0 |         223 |         65.5% |
| NEAR  |            514 / 2,190 |       204 |        109 |                   11 |       2,281 |         21.6% |
| UNI   |              260 / 799 |        45 |         17 |                    1 |         508 |         11.8% |

$/day figures are collection-level rates (total ÷ 268.2 clean h × 24). Previously
reported medians/totals were: BTC 222/$1,689, ARB 27/$34, NEAR 231/$634, UNI
260/$508 — UNI is unchanged to the cent because its ctVal is 1, which is the
expected check that the fix touches only non-unit contract values. The Phase B
conclusion (BTC control failed via stale-quote artifact; 87–97% one-sided direction
= structural premium) does not depend on these magnitudes and stands.

---
