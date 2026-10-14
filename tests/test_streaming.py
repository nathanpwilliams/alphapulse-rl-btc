"""
tests/test_streaming.py - Unit tests for
src/strategies/dislocation/streaming.py

Synthetic multi-day top-of-book CSVs written to tmp_path; no network. Guards:
the chunked per-(day, asset) read keeps only the requested asset and is
chunksize-invariant, mask windows drop rows AND discount active seconds,
per-excursion records are evaluated at the peak-net row (capacity, net bps and
direction all read from the same row), the day-by-day driver reproduces the
single-frame analyzer's excursion count, and a flat/efficient asset yields zero
excursions (the BTC-control shape).

Run:
    pytest tests/test_streaming.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.collect_top_of_book import TOB_COLUMNS  # noqa: E402
from src.strategies.dislocation import executable as ex  # noqa: E402
from src.strategies.dislocation import streaming as st  # noqa: E402


# ============================================================
# Builders
# ============================================================

def _row(ts_ms: int, venue: str, asset: str, bid: float, bid_sz: float,
         ask: float, ask_sz: float) -> dict:
    return {"ts_recv": ts_ms, "ts_exchange": ts_ms - 20, "venue": venue,
            "asset": asset, "bid": bid, "bid_size": bid_sz,
            "ask": ask, "ask_size": ask_sz}


def _day_ms(day: str) -> int:
    """UTC midnight of 'YYYYMMDD' in epoch ms."""
    return int(pd.Timestamp(day, tz="UTC").value // 1_000_000)


def write_day(
    out_dir: Path, day: str, rows: list[dict],
) -> Path:
    """Write a synthetic `tob_<day>.csv` with the collector's exact schema."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tob_{day}.csv"
    pd.DataFrame(rows, columns=TOB_COLUMNS).to_csv(path, index=False)
    return path


def flat_rows(day: str, asset: str, n: int = 40, step_ms: int = 100,
              price: float = 100.0, size: float = 10.0) -> list[dict]:
    """Both venues quoting an identical tight book: no cross-venue edge at all."""
    t0 = _day_ms(day)
    rows = []
    for i in range(n):
        ts = t0 + i * step_ms
        for venue in ("kraken", "okx"):
            rows.append(_row(ts, venue, asset, price - 0.01, size, price + 0.01, size))
    return rows


def dislocated_rows(day: str, asset: str, at: list[int], n: int = 40,
                    step_ms: int = 100, price: float = 100.0) -> list[dict]:
    """Flat book except at update indices `at`, where Kraken's bid is far above
    OKX's ask (a big sell-Kraken/buy-OKX edge, clears 10 bps round-trip)."""
    t0 = _day_ms(day)
    rows = []
    for i in range(n):
        ts = t0 + i * step_ms
        if i in at:
            # kraken bid 101.0 vs okx ask 100.01 -> ~98 bps gross, clears fees.
            rows.append(_row(ts, "kraken", asset, 101.00, 3.0, 101.02, 7.0))
            rows.append(_row(ts, "okx", asset, 99.99, 9.0, 100.01, 5.0))
        else:
            rows.append(_row(ts, "kraken", asset, price - 0.01, 10.0, price + 0.01, 10.0))
            rows.append(_row(ts, "okx", asset, price - 0.01, 10.0, price + 0.01, 10.0))
    return rows


def _cfg(**kw) -> st.StreamConfig:
    kw.setdefault("executable", ex.ExecutableConfig(staleness_tolerance="500ms"))
    return st.StreamConfig(**kw)


# ============================================================
# Chunked loading
# ============================================================

def test_load_asset_day_keeps_only_requested_asset(tmp_path):
    rows = flat_rows("20261001", "BTC") + flat_rows("20261001", "ARB", price=2.0)
    path = write_day(tmp_path, "20261001", rows)

    k, o = st.load_asset_day(path, "ARB", _cfg())

    assert len(k) == 40 and len(o) == 40
    # ARB was written at price 2.0; BTC rows must not leak in.
    assert np.allclose(k["bid"].to_numpy(), 1.99)
    assert list(k.columns) == st.QUOTE_COLS
    assert isinstance(k.index, pd.DatetimeIndex) and str(k.index.tz) == "UTC"


def test_load_asset_day_is_chunksize_invariant(tmp_path):
    rows = dislocated_rows("20261001", "ARB", at=[10, 11]) + flat_rows("20261001", "BTC")
    path = write_day(tmp_path, "20261001", rows)

    big = st.load_asset_day(path, "ARB", _cfg(chunksize=1_000_000))
    small = st.load_asset_day(path, "ARB", _cfg(chunksize=7))  # forces many chunks

    for a, b in zip(big, small):
        pd.testing.assert_frame_equal(a, b)


def test_load_asset_day_missing_asset_returns_empty(tmp_path):
    path = write_day(tmp_path, "20261001", flat_rows("20261001", "BTC"))

    k, o = st.load_asset_day(path, "DOGE", _cfg())

    assert k.empty and o.empty
    assert list(k.columns) == st.QUOTE_COLS


def test_mask_window_drops_rows(tmp_path):
    path = write_day(tmp_path, "20261001", dislocated_rows("20261001", "ARB", at=[10]))
    t0 = pd.Timestamp("20261001", tz="UTC")
    # Mask updates 5..14 (step 100ms), which contains the only excursion.
    cfg = _cfg(mask_windows=[(t0 + pd.Timedelta("500ms"), t0 + pd.Timedelta("1500ms"))])

    k, o = st.load_asset_day(path, "ARB", cfg)

    assert len(k) == 30 and len(o) == 30
    assert not ((k.index >= t0 + pd.Timedelta("500ms"))
                & (k.index < t0 + pd.Timedelta("1500ms"))).any()
    recs, _ = st.asset_day_records(k, o, cfg)
    assert recs == []          # the dislocation was inside the masked window


# ============================================================
# mask_seconds
# ============================================================

@pytest.mark.parametrize("w0,w1,expected", [
    ("2026-10-01 00:00", "2026-10-01 01:00", 3600.0),   # fully inside
    ("2026-09-30 23:00", "2026-10-01 01:00", 3600.0),   # clipped at the left
    ("2026-10-01 23:00", "2026-10-02 05:00", 3600.0),   # clipped at the right
    ("2026-10-05 00:00", "2026-10-05 01:00", 0.0),      # no overlap
])
def test_mask_seconds_clips_to_slice(w0, w1, expected):
    start = pd.Timestamp("2026-10-01 00:00", tz="UTC")
    end = pd.Timestamp("2026-10-02 00:00", tz="UTC")
    windows = [(pd.Timestamp(w0, tz="UTC"), pd.Timestamp(w1, tz="UTC"))]

    assert st.mask_seconds(start, end, windows) == pytest.approx(expected)


def test_mask_seconds_discounts_active_time(tmp_path):
    path = write_day(tmp_path, "20261001", dislocated_rows("20261001", "ARB", at=[10], n=40))
    t0 = pd.Timestamp("20261001", tz="UTC")
    cfg = _cfg(mask_windows=[(t0 + pd.Timedelta("3000ms"), t0 + pd.Timedelta("3500ms"))])

    k, o = st.load_asset_day(path, "ARB", cfg)
    _, stats = st.asset_day_records(k, o, cfg)

    # Slice spans ~3.9s; 0.5s of it is masked out of the active denominator.
    assert stats["masked_seconds"] == pytest.approx(0.5)
    assert stats["active_seconds"] == pytest.approx(stats["covered_seconds"] - 0.5)


# ============================================================
# Per-excursion records
# ============================================================

def test_records_evaluated_at_peak_row(tmp_path):
    path = write_day(tmp_path, "20261001", dislocated_rows("20261001", "ARB", at=[10, 11]))
    k, o = st.load_asset_day(path, "ARB", _cfg())

    recs, stats = st.asset_day_records(k, o, _cfg())

    assert len(recs) == 1
    r = recs[0]
    assert r["direction"] == "sell_k_buy_o"
    # mid = (101.00 + 101.02 + 99.99 + 100.01)/4 = 100.505
    assert r["mid"] == pytest.approx(100.505, abs=1e-6)
    # gross = 10000*(101.00 - 100.01)/100.505 = 98.50 bps; net = -10 fees.
    assert r["net_bps"] == pytest.approx(98.50 - 10.0, abs=0.05)
    # capacity = min(k_bid_size=3, o_ask_size=5) * mid = 3 * 100.505
    assert r["cap_usd"] == pytest.approx(3 * 100.505, abs=1e-3)
    assert r["capturable_usd"] == pytest.approx(r["cap_usd"] * r["net_bps"] / 10000.0)
    assert stats["best_net_bps"] == pytest.approx(r["net_bps"], abs=0.05)
    assert r["hour"] == 0 and r["weekday"] == pd.Timestamp("20261001", tz="UTC").weekday()


def test_flat_book_yields_no_excursions(tmp_path):
    """The BTC-control shape: identical books on both venues -> nothing clears fees."""
    path = write_day(tmp_path, "20261001", flat_rows("20261001", "BTC"))
    k, o = st.load_asset_day(path, "BTC", _cfg())

    recs, stats = st.asset_day_records(k, o, _cfg())

    assert recs == []
    # best possible is -0.02/100 *10000 - 10 fees, comfortably negative.
    assert stats["best_net_bps"] < 0
    assert stats["n_aligned"] > 0


def test_two_separated_dislocations_are_two_excursions(tmp_path):
    path = write_day(tmp_path, "20261001", dislocated_rows("20261001", "ARB", at=[8, 25]))
    k, o = st.load_asset_day(path, "ARB", _cfg())

    recs, _ = st.asset_day_records(k, o, _cfg())

    assert len(recs) == 2
    assert all(r["duration_sec"] > 0 for r in recs)


def test_short_slice_returns_empty_stats(tmp_path):
    path = write_day(tmp_path, "20261001", flat_rows("20261001", "ARB", n=1))
    k, o = st.load_asset_day(path, "ARB", _cfg())

    recs, stats = st.asset_day_records(k, o, _cfg())

    assert recs == [] and stats["n_aligned"] == 0


# ============================================================
# Multi-day driver
# ============================================================

def test_analyze_streaming_spans_days(tmp_path):
    write_day(tmp_path, "20261001", dislocated_rows("20261001", "ARB", at=[10]))
    write_day(tmp_path, "20261002", dislocated_rows("20261002", "ARB", at=[5, 20]))
    write_day(tmp_path, "20261003", flat_rows("20261003", "ARB"))

    res = st.analyze_streaming(tmp_path, "ARB", _cfg())

    assert sorted(res["per_day"]) == ["20261001", "20261002", "20261003"]
    assert res["per_day"]["20261001"]["n_excursions"] == 1
    assert res["per_day"]["20261002"]["n_excursions"] == 2
    assert res["per_day"]["20261003"]["n_excursions"] == 0
    assert len(res["records"]) == 3
    assert {r["day"] for r in res["records"]} == {"20261001", "20261002"}
    assert res["per_day"]["20261002"]["capturable_usd"] == pytest.approx(
        sum(r["capturable_usd"] for r in res["records"] if r["day"] == "20261002"))


def test_streaming_matches_single_frame_analyzer(tmp_path):
    """The whole point: driving day-by-day must not change the analyzer's answer."""
    write_day(tmp_path, "20261001", dislocated_rows("20261001", "ARB", at=[10]))
    write_day(tmp_path, "20261002", dislocated_rows("20261002", "ARB", at=[5, 20]))

    res = st.analyze_streaming(tmp_path, "ARB", _cfg())
    streamed = len(res["records"])

    # Single-frame path over the same files (what load_tob would do in-memory).
    k, o = ex.load_tob(tmp_path, "ARB")
    single = ex.analyze_executable(k, o, ex.ExecutableConfig(staleness_tolerance="500ms"))

    assert streamed == single["executable"]["n_excursions"] == 3
    assert max(d["best_net_bps"] for d in res["per_day"].values()) == pytest.approx(
        single["net_gap_bps"]["max"], abs=1e-6)


def test_daily_files_excludes_events_log(tmp_path):
    write_day(tmp_path, "20261001", flat_rows("20261001", "ARB"))
    (tmp_path / "tob_events_20261001.csv").write_text("ts_recv,venue,asset,event,detail\n")

    files = st.daily_files(tmp_path)

    assert [f.name for f in files] == ["tob_20261001.csv"]


# ============================================================
# Size units threaded through the driver
# ============================================================

def test_asset_resolves_the_okx_contract_multiplier(tmp_path):
    """A ctVal > 1 enlarges the OKX leg, so a Kraken-bound cap is unaffected."""
    write_day(tmp_path, "20261001", dislocated_rows("20261001", "NEAR", at=[10]))
    k, o = st.load_asset_day(tmp_path / "tob_20261001.csv", "NEAR", _cfg())

    without = st.asset_day_records(k, o, _cfg())[0]          # multipliers 1.0/1.0
    with_asset = st.asset_day_records(k, o, _cfg(), "NEAR")[0]   # OKX ctVal = 10

    assert len(without) == len(with_asset) == 1
    # sell-kraken/buy-okx: cap = min(k_bid 3, o_ask 5 x mult) x mid.
    # mult=1 -> min(3, 5)=3; mult=10 -> min(3, 50)=3. Kraken binds either way.
    assert without[0]["okx_leg_binds"] is False
    assert with_asset[0]["okx_leg_binds"] is False
    assert with_asset[0]["cap_usd"] == pytest.approx(without[0]["cap_usd"])
    assert with_asset[0]["cap_usd"] == pytest.approx(3.0 * with_asset[0]["mid"], rel=1e-9)


def test_contract_multiplier_changes_capacity_when_okx_binds(tmp_path):
    """BTC ctVal 0.01 shrinks the OKX leg 100x, so it becomes the binding side."""
    rows = dislocated_rows("20261001", "BTC", at=[10])
    write_day(tmp_path, "20261001", rows)
    k, o = st.load_asset_day(tmp_path / "tob_20261001.csv", "BTC", _cfg())

    plain = st.asset_day_records(k, o, _cfg())[0]
    btc = st.asset_day_records(k, o, _cfg(), "BTC")[0]

    assert btc[0]["cap_usd"] < plain[0]["cap_usd"]
    assert btc[0]["okx_leg_binds"] is True
    # cap scales with the OKX leg: 5 contracts x 0.01 = 0.05 base units.
    assert btc[0]["cap_usd"] == pytest.approx(0.05 * btc[0]["mid"], rel=1e-9)


def test_analyze_streaming_passes_the_asset_through(tmp_path):
    write_day(tmp_path, "20261001", dislocated_rows("20261001", "BTC", at=[10]))

    res = st.analyze_streaming(tmp_path, "BTC", _cfg())

    assert len(res["records"]) == 1
    r = res["records"][0]
    assert r["okx_leg_binds"] is True
    assert r["cap_usd"] == pytest.approx(0.05 * r["mid"], rel=1e-9)
