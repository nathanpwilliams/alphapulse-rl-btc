# CLAUDE.md

Project context and conventions for the **AlphaPulse RL BTC** codebase. Read this
before adding code so new work stays consistent with what's already here.

## What this project is

A crypto quant research platform for BTC and liquid alts, structured like a
miniature [Qlib](https://github.com/microsoft/qlib): clean separation between
data loading, the trading environment, model training, and strategy modules. The
research question is whether a tradable edge survives realistic transaction
costs. Six candidates were tested and all six rejected on evidence; `README.md`
has the results table and `docs/findings.md` the full write-up.

## Architecture

Source lives under `src/`, one concern per package:

| Package                        | Responsibility                                                                                                                                                    |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/data/`                    | Data loaders — `load_kraken.py` (Kraken Futures OHLCV + funding), `load_okx.py`, `load_cross_venue.py`, `collect_top_of_book.py` (live websocket quote collector) |
| `src/env/`                     | Gymnasium RL trading environment (`trading_env.py`), optional regime-posterior state                                                                              |
| `src/training/`                | PPO training scaffold (`train.py`) with chronological validation callback                                                                                         |
| `src/strategies/funding/`      | Funding-rate arb — `analyzer.py` (single-venue), `cross_venue.py` (cadence-normalized Kraken–OKX spread)                                                         |
| `src/strategies/regime/`       | HMM regime engine — `detector.py`, `features.py` (causal walk-forward posteriors), `backtest.py` (walk-forward filter)                                            |
| `src/strategies/dislocation/`  | Cross-venue dislocation — `scanner.py` (Phase A candles), `executable.py` (Phase B quotes), `streaming.py`, `report.py`                                           |
| `src/strategies/mm/`           | `spread_feasibility.py` — passive spread-capture economics                                                                                                        |
| `src/backtest/`, `src/agents/` | Reserved package stubs; no modules yet                                                                                                                            |

Tests mirror the source tree under `tests/` (`tests/test_data.py`,
`tests/test_funding_analyzer.py`, ...) — 327 tests at last count.

## Coding conventions

Match the style of `src/data/load_kraken.py`, which is the reference module:

- **Python 3.9 compatible.** The interpreter is 3.9.6. Put
  `from __future__ import annotations` at the top of every module so that PEP 604
  (`int | None`) and builtin-generic (`list[dict]`) type hints work under 3.9.
- **Type hints throughout** — annotate public function signatures and returns.
- **`dataclass` for configs** — group tunable parameters into a
  `@dataclass` config object rather than long keyword argument lists.
- **Module-level logger** — `logger = logging.getLogger(__name__)` at module top;
  never call `logging.basicConfig` except inside a CLI entrypoint.
- **Docstrings on public functions** — explain args, returns, and any non-obvious
  units or invariants. Google-style arg blocks as in `load_kraken.py`.
- **Explicit units.** Financial math is unit-sensitive (per-hour rate → bps →
  annualized). Annotate every conversion inline; this is the easiest place to
  introduce a silent bug.
- **Section banners** — the `# ===...===` comment banners separating logical
  sections are the house style; keep using them.
- **CLI smoke test** — modules that touch live data expose a `main()` / smoke test
  runnable as `python -m src.<pkg>.<module>`.

## Testing discipline

- **pytest**, tests in `tests/`, every module gets a matching test file.
- **Prefer synthetic-data unit tests that don't hit the network.** Construct
  DataFrames in-test or mock `_http_get_json`; live-data correctness is left to the
  module's CLI smoke test. See `tests/test_data.py` for the pattern (synthetic
  Kraken response builders, `unittest.mock.patch`).
- Tests insert the project root on `sys.path` and import as `from src.<pkg> ...`.
- Run the suite with `pytest tests/ -v`.

## Key project findings

Six candidate edges were tested and all six rejected on evidence, each with a
mechanism (full detail in `docs/findings.md`):

1. **RL directional** — overfits; data volume is the binding constraint, not
   model capacity (~10 months of BTC 4H, 7,123 train bars).
2. **Funding arb** (single + cross-venue) — sign flips every ~10 h; break-even
   needs ~70 h even at 5 bps round trip.
3. **Regime filter** — Sharpe −0.392 vs buy-and-hold +1.130; detection lag plus a
   rule that filters out the trend.
4. **Regime-conditional RL** — train/val gap widened 0.127 → 0.296; added input
   dims become overfit capacity on limited data.
5. **Cross-venue alt dislocation** — rejected by its own BTC control: Kraken's
   ~1 s snapshot feed vs OKX's tick stream manufactures phantom sub-second gaps.
6. **Passive spread capture** — every edge ratio below 1.0 (max observed 0.57);
   adverse selection scales with spread width.

The platform and the methodology are the deliverable, not a strategy.

## Journal policy

`docs/journal/project_journal.md` is a **technical record only** — decisions,
experiments, findings, lessons. It is public. Record technical decisions and
results only; no personal or career material. That lives in a private file
outside the project.

## Commit policy

Commit messages in this repo carry **no AI co-authorship trailers** — no
`Co-Authored-By:` and no "Generated with" lines. Prose in docs that describes the
tooling honestly is fine and should not be scrubbed.

## Cost realism principle

**Transaction costs and honest accounting are non-negotiable.** A strategy must
clear realistic **round-trip** costs (both legs, entry + exit, including slippage)
before it is considered viable. Analysis code should make the cost assumption an
explicit, sweepable parameter and report net-after-cost figures, never gross-only.
