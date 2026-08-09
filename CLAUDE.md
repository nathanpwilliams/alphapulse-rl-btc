# CLAUDE.md

Project context and conventions for the **AlphaPulse RL BTC** codebase. Read this
before adding code so new work stays consistent with what's already here.

## What this project is

A crypto quant research platform for BTC, structured like a miniature
[Qlib](https://github.com/microsoft/qlib): clean separation between data loading,
the trading environment, model training, and (now) strategy modules. The research
question is whether a tradable BTC edge survives realistic transaction costs.

## Architecture

Source lives under `src/`, one concern per package:

| Package          | Responsibility                                                        |
| ---------------- | -------------------------------------------------------------------- |
| `src/data/`      | Data loaders (e.g. `load_kraken.py` — Kraken Futures OHLCV + funding) |
| `src/env/`       | Gymnasium RL trading environment (`trading_env.py`)                   |
| `src/training/`  | PPO training scaffold (`train.py`)                                    |
| `src/backtest/`  | Backtesting utilities                                                 |
| `src/agents/`    | Agent definitions                                                     |
| `src/strategies/`| Strategy modules (new — e.g. funding-rate arbitrage analysis)        |

Tests mirror the source tree under `tests/` (`tests/test_data.py`,
`tests/test_funding_analyzer.py`, ...).

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

## Key project finding

RL **directional** trading on ~1 year of BTC 4H data **overfits** — data volume is
the binding constraint, not model capacity. Funding-rate arbitrage is being
explored as a **structural (non-predictive) edge** that sidesteps the overfitting
problem: it earns from a market-structure spread rather than from forecasting price.

## Cost realism principle

**Transaction costs and honest accounting are non-negotiable.** A strategy must
clear realistic **round-trip** costs (both legs, entry + exit, including slippage)
before it is considered viable. Analysis code should make the cost assumption an
explicit, sweepable parameter and report net-after-cost figures, never gross-only.
