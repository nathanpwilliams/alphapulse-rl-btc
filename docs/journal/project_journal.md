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