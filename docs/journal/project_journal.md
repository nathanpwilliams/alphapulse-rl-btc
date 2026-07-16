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