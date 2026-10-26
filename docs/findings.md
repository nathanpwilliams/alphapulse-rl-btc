# Findings

What this project set out to do, what it measured, and why each of six candidate
edges was rejected. Every figure here comes from the project journal
(`docs/journal/project_journal.md`), which records the runs as they happened,
including the superseded numbers from one unit-error correction.

The short version: across six strategies on two centralized venues, nothing
cleared its pre-registered bar. The useful output is not a strategy but a map of
*where* retail-reachable edge is absent and *which mechanism* removes it in each
case — plus a platform whose guards make a negative result believable.

---

## 1. The data constraint

**Hypothesis.** A PPO agent on BTC 4H bars can learn a directional policy that
generalizes to a later, held-out period.

**Measurement.** The training scaffold evaluates the policy every 10K steps on a
chronologically separate validation slice and checkpoints best-by-validation
independently from the final model. The comparison that matters is the *shape* of
train versus validation, not the level of either.

**Pre-registered bar.** Validation reward must rise, or at least hold, while
training reward rises. Divergence is overfitting, regardless of how good the
training curve looks.

**Result.** The first run diverged immediately: training reward rose while
validation reward declined into negative territory (~−0.03) from roughly step
10K. Three single-variable experiments followed, each changing exactly one thing
and holding seed, environment count and hyperparameters fixed.

Widening the data window from ~9 to ~12 months (the full Kraken funding history —
8,838 hourly rates, 2025-07-13 → 2026-07-16) measurably reduced the divergence:
validation reward stayed small-but-positive (~+0.005) with no sustained downward
trend. Scaling timesteps 57K → 500K then *fixed the critic* —
`explained_variance` crossed from ~−12 to ~+0.39 — but *deepened the overfit*:
validation peaked around step 50–70K at ~+0.007 and fell to ~−0.033 while
training reward climbed to +0.09. Reducing network capacity from [64,64] to
[32,32] softened the mid-training trough and slightly improved the critic (0.485
vs 0.442 final explained_variance) but both architectures converged to the same
poor endpoint, validation ~−0.04 to −0.047 at 500K.

**Mechanism.** Three experiments pointing three different levers at the same
problem all landed on one constraint: **data volume**. Roughly ten months of BTC
4H history — 7,123 training bars — does not contain enough distinct market
regimes for a policy to generalize to a held-out period. Architecture and
training duration change the shape of the overfit, not its existence. The
training reward curve is actively misleading here: +0.09 looked like profitable
learning and was memorization. Only the validation curve told the truth.

---

## 2. Funding arbitrage: a structural edge with the wrong clock

**Hypothesis.** If prediction is the problem, stop predicting. A delta-neutral
short-perp / long-spot position receives funding whenever funding is positive —
income from market structure rather than from a forecast.

**Measurement.** Kraken's `funding_rate` is a per-hour proportional rate. The
analyzer converts it to basis points and annualized terms with every conversion
annotated inline, then asks the only question that matters: does funding
collected over a realistic holding period exceed the all-in round-trip cost of
opening and closing both legs?

**Pre-registered bar.** Net-after-cost must be positive at a realistic
round-trip cost, with the cost an explicit sweepable parameter rather than an
assumption buried in the code.

**Result.** Over 8,818 hourly Kraken PF_XBTUSD rates (~368 days), funding was
positive 71.9% of the time and averaged +3.77%/yr signed, +6.30%/yr absolute —
real income. But the sign flips in 9.84% of intervals, roughly every 10.2 hours,
while break-even requires a ~278-hour hold at 20 bps round trip. A cost sweep at
5/10/15/20/25 bps gives 2.9 / 5.8 / 8.7 / 11.6 / 14.5 days: even at an aggressive
5 bps floor the hold is ~70 hours. Net-after-cost is negative at 1-, 3- and
7-day holds and only marginally positive at 14–30 days.

**Mechanism.** The income's sign-change cadence is roughly an order of magnitude
shorter than the hold needed to pay for entry. The position is forced to close —
or to flip from receiving to paying — long before it has earned its transaction
costs. This is not a cost-tuning problem; it is a timescale mismatch.

The natural follow-up was the funding *spread* between two venues, which only
needs the spread to be stable rather than either venue's rate. That analysis
required getting one thing right: Kraken settles funding hourly, while OKX
settles on a longer, dynamic interval (8 h at the time, historically 4 h,
escalating under stress) and publishes a *period* rate — the loader infers the
interval from timestamps rather than assuming it. Subtracting an OKX period rate
from a Kraken hourly rate would have been a correctness bug invalidating every
downstream number, so both venues are normalized to a common per-hour basis
before any spread is taken, aligned on an hourly grid with a per-venue staleness
cap so a data gap cannot propagate an ancient rate.

The spread was worse, not better. Over the 92-day overlap (OKX's API depth limit)
the spread's sign flips every ~6.1 hours against the single venue's ~10.2, and
the two venues' funding correlates at only **+0.145**. Mean |spread| is 0.0705
bps/hr against Kraken's own mean |rate| of 0.0720 — the spread is essentially
Kraken's funding plus noise, not a stable residual. A harvestable cross-venue
spread needs highly correlated venues leaving a small, stable difference; at
+0.15, differencing compounds noise instead of cancelling it.

---

## 3. Regimes: real, and not tradeable this way

**Hypothesis.** BTC has distinguishable volatility/drift regimes. If so, either
filtering by regime or feeding regime to the agent should help.

**Measurement.** An HMM detector with per-asset AIC/BIC state selection — no
hardcoded regime count — chose 4 states. The critical design choice was
walk-forward labeling: the HMM is refit on a rolling cadence, the z-score scaler
is split so test bars are standardized with *training-window statistics only*,
each bar is labeled from its past alone, and the strategy acts on the *prior*
bar's regime to model detection lag.

That honesty is expensive and it is the point. Walk-forward labels are far more
conservative than an in-sample Viterbi decode: the regime known only from the
past is lagged and noisier than the full-sample path. An in-sample-labelled
backtest would have looked much better and meant nothing.

### Level 2 — regime as a filter

**Pre-registered bar.** Improve risk-adjusted return versus buy-and-hold, with
honest cost and detection-lag accounting.

**Result.** Live PF_XBTUSD 4H, 2022-11-30 → 2026-08-21, 8,165 bars, 46
walk-forward refits, 10 bps/side:

| Metric | Regime filter | Buy & hold |
| --- | --- | --- |
| Annualized return | −10.59% | +51.01% |
| Annualized vol | 22.25% | 45.73% |
| Sharpe (rf=0) | **−0.392** | **+1.130** |
| Max drawdown | −51.58% | −53.55% |
| Calmar | −0.205 | +0.952 |

**Mechanism.** The "positive drift, below-median vol" favorable rule kept the
book in cash 68% of the time, and much of BTC's return over this window arrived
in exactly the bars the rule flags unfavorable — higher-volatility up-moves.
Filtering those out halved volatility and surrendered the trend with it.
Drawdown protection was negligible (−51.6% vs −53.6%) despite the low exposure:
with one-bar detection lag and a 180-bar refit cadence, the filter re-enters into
and holds through declines rather than avoiding them. Turnover was modest — 592
position changes over 8,165 bars — so transaction cost is not the driver. The
long/flat signal itself is.

### Level 3 — regime as a causal state feature

**Pre-registered bar.** The regime-augmented agent must generalize better than a
baseline that is identical in every other respect.

**Result.** Both runs at 500K steps, 8 envs, seed 42, [32,32],
2022-04-01 → 2026-08-01, 7,123 train / 2,375 val bars — only the feature differs:

| Metric | Baseline | Regime-augmented |
| --- | --- | --- |
| Final train reward | +0.0817 | +0.2126 |
| Final val reward | −0.0459 | −0.0837 |
| Peak val reward | +0.0097 (410K) | +0.0078 (450K) |
| Train/val gap | **0.127** | **0.296** |

**Mechanism.** The 4-dimensional regime posterior gave the policy extra input
capacity, which on a data-limited problem it spent memorizing the training
period: training reward nearly tripled while validation degraded. This is the
same binding constraint as section 1, arriving through a different door — more
input dimensions means more overfitting capacity when data is the limit.

One detail is worth stating because it is the project's cleanest piece of
evidence *for* its own methodology: the augmented agent looked **worse** on
validation, not suspiciously better. A regime feature leaking future information
would have inflated validation reward. Instead the causal walk-forward posterior
plus one-bar lag produced a feature that simply did not help. That the result is
unimpressive is itself evidence the causality is clean. (The soft posterior's
causality rests on a clean property: the forward-backward marginal at a
sequence's last position equals the filtered posterior, so
`predict_proba(window)[-1]` uses no data beyond the window end.)

---

## 4. The pivot: a less efficient arena

Four strategies had now failed with one common cause — an efficient,
data-limited arena probed with predictive edges. The response was to change
arena rather than model: look for a *structural* dislocation where liquidity is
thinner. Cross-venue price gaps between Kraken Futures and OKX across a basket of
alts, with BTC and ETH carried along as efficient-market controls.

Probing the venues first established a constraint: 173 shared base assets, but
Kraken Futures perps are USD-quoted and OKX USDT-quoted, so same-quote pairing is
infeasible. The basis is therefore left to ride as a structural premium that the
analyzers separate from transient dislocation.

### Phase A — the candle screen

**Result (30 days, 5m bars, 20 bps hurdle).** Controls clean: BTC and ETH showed
~0.1 transient excursions/day, mean |gap| ~3 bps, half-life 18–34 min. Alts
showed a monotonic liquidity gradient — UNI 7.4, NEAR 17.9, ARB 41.9 transient
excursions/day; ARB mean |gap| 15.5 bps, p95 38.5 bps, half-life 1.6 min.
Structural premium positive on every asset and rising with illiquidity (BTC +2.4
→ ARB +9.4 bps).

This was the first positive-direction confirmation in the project: no exploitable
gap where the market is efficient, rising dislocation as liquidity thins — exactly
what theory predicts. It was also immediately disbelieved, for three reasons
recorded at the time: ARB's 1.6-minute half-life is *shorter than the 5-minute
sampling interval*, so most of an excursion reverts between observations; candle
closes are last-trade prints that go stale on thin books and manufacture phantom
gaps; and a 20 bps hurdle assumes a spread the candles cannot see. All three are
resolvable only with bid/ask quotes.

### Phase B — quotes, and the control that failed

**Pre-registered bar,** fixed before any data existed: BTC control excursions
~0; a >1 s duration filter to clear Kraken's feed resolution; directional
symmetry.

**Measurement.** A websocket collector ran as a systemd service under an
unprivileged user on a dedicated droplet for 11.2 days: 71.5M rows, 4.4 GB, 268.2
clean hours, zero service restarts, one 397 s Kraken outage masked rather than
interpolated. Executable gaps only — bid on one venue against ask on the other,
net of fees — computed through a chunked streaming driver (peak RSS 0.78 GB)
around the unmodified analyzer.

**Result.** Every criterion failed. The BTC control produced 276 net-positive
excursions, best +41 bps, 88.8% of them lasting ≤1.0 s, where the bar required
~0. Alts produced ARB 272, NEAR 527, UNI 508 excursions/day. Direction was 87–97%
one-sided (sell-Kraken / buy-OKX), persistently, across all 12 days. Restricting
to transient excursions longer than 1 s, the corrected capturable figures are
BTC $13, NEAR $11, UNI $1 and ARB $0 per day.

**Mechanism.** The control failing *is* the result. Kraken's ~1 s snapshot feed
against OKX's tick-by-tick stream manufactures sub-second phantom gaps during
fast moves — a stale-quote illusion. Apparent dollars scale with volatility
because they are latency, not mispricing. Without the control, NEAR and UNI would
have read as a ~$50/day edge. The persistent one-sided component is the USD/USDT
quote-currency premium leaking through raw gaps; it has no closing leg without
paying the transfer friction that sustains it, so it is not an arbitrage. Phase
A's liquidity ordering did not reproduce at quote resolution — ARB turned out to
have the least capacity and the least money.

A unit error was found and published during this phase: OKX quotes size in
contracts and Kraken in base units, so `min(bid_size, ask_size)` mixed units. The
corrected figures (ctVal: BTC 0.01, NEAR 10, UNI 1, ARB 10) sit in the journal
next to the superseded ones, with UNI unchanged to the cent because its ctVal is
1 — the expected check that the fix touched only non-unit contracts. The Phase B
conclusion never depended on those magnitudes.

---

## 5. Passive spread capture

The thin-alt bid-ask spread that made the arbitrage uncapturable was the last
income source on the original list not yet measured. If the spread is too wide to
cross, perhaps it can be *earned*.

**Measurement.** A data-only analysis on the existing 11-day quote set: spread
distributions, net half-spread after base-tier maker fees (2 bps/side on both
venues), realized volatility at 1/5/30 s, and a naive touch-quoter adverse-event
model treating a trade-through as a certain adverse fill, with markouts.

**Pre-registered bar.** Edge ratio — net half-spread ÷ mean adverse markout at
τ=1 s — below 1 on both pre-registered alts means dead.

**Result.** Dead on both, on both venues, in every hour. ARB Kraken 0.57 and
NEAR Kraken 0.38 are the highest ratios observed anywhere; UNI Kraken −0.13; OKX
negative throughout (−0.47 NEAR, −0.91 UNI, −1.10 ARB). The BTC control came in
at −3.2 / −2.5, correctly unmakeable at 1 s. No hour-of-day on any alt exceeds
0.45; weekends cut adverse rates 30–60% and leave the ratios unchanged.

**Mechanism.** Two distinct failure modes. On OKX, professional makers already
quote alt spreads around 2.0 bps — below a retail maker's 4 bps round-trip fee,
100% of the time. That is a fee-tier and latency arms race, not a spread to earn.
On Kraken, spreads are wide (NEAR median 8.8 bps, net half +2.4) but adverse
selection at 1 s costs about 2.5× the net half-spread (6.4). Spread width and
short-horizon adverse moves scale together, which is why no venue, hour, or
weekday/weekend split clears 1. And because Kraken's snapshot feed undercounts
adverse events — 43/hour against OKX's 1,123/hour for the same asset at the same
realized volatility — even those Kraken ratios are upper bounds.

---

## The empirical picture

Six strategies, six rejections, each with a mechanism rather than a shrug:

1. **RL directional** — overfits; data volume binding.
2. **Funding arb** — sign-flip cadence ~7× shorter than break-even hold.
3. **Regime filter** — detection lag plus a rule that filters out the trend.
4. **Regime-conditional RL** — added input dimensions become overfit capacity.
5. **Cross-venue dislocation** — latency artifact plus an unclosable quote premium.
6. **Passive spread capture** — adverse selection scales with spread width.

The common thread is sharper than "markets are efficient." Edges reachable from
retail infrastructure on centralized venues are already captured by faster
capital. The sub-second excursions recorded in Phase B are real events — they are
other participants' trades, visible after the fact through a feed slow enough to
make them look available. Two of the six rejections (5 and 6) turn specifically
on *measurement* resolution rather than on the strategy idea, and would have read
as positive results on a platform without controls and duration filters.

What survives is the apparatus: pre-registered bars that make a marginal result
hard to rationalize, controls permitted to fail and allowed to overturn the
analysis when they do, look-ahead guards enforced as non-negotiable tests rather
than asserted in comments, round-trip costs as a sweepable parameter with
net-only reporting, single-variable experiments that keep attribution clean, and
corrections published next to the figures they replace. 327 tests hold those
properties in place.

On-chain and DEX venues have different microstructure, different participants,
and do not share the latency asymmetry that decided section 4. Whether edge
survives costs there is **untested here** — and saying so is a statement about
the boundary of this evidence, not a plan.
