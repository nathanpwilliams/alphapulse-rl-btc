"""
src/data/load_cross_venue.py - Multi-asset, two-venue OHLCV loader.

Bridges the Kraken Futures loader (`fetch_ohlcv`, USD-quoted PF_ perps) and the
OKX candle loader (`fetch_okx_ohlcv`, USDT-quoted swaps) to pull the same asset
from both venues for cross-venue price-dislocation analysis.

QUOTE-CURRENCY NOTE (important, documented once here): Kraken Futures perpetuals
are all USD-quoted (`PF_<BASE>USD`, multi-collateral); OKX swaps are USDT-quoted
(`<BASE>-USDT-SWAP`). Kraken Futures does not offer USDT-quoted perps, so a
USDT-on-both basket is impossible. The residual USD/USDT basis (usually a few bps)
therefore rides on every cross-venue gap as a PERSISTENT premium - the dislocation
analyzer separates that structural component (mean signed gap + sign stability)
from the transient, mean-removed excursions that are the actual arbitrage signal.

The default basket (10 assets) was chosen from the live both-venue intersection:
BTC and ETH as efficient-market controls, plus 8 alts spanning liquidity tiers
(~$1.2B down to ~$0.1B OKX 24h notional at selection time).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from src.data.load_kraken import DEFAULT_CACHE_DIR, PER_REQUEST_DELAY_SEC, fetch_ohlcv
from src.data.load_okx import fetch_okx_ohlcv
from src.data.load_okx import _http_get_json  # reused hardened GET for ticker liquidity

logger = logging.getLogger(__name__)


# ============================================================
# Basket
# ============================================================
#
# (base, kraken_symbol, okx_inst_id). BTC/ETH are the controls; the rest span
# liquidity tiers. See module docstring for the USD-vs-USDT quote caveat.

DEFAULT_BASKET: list[tuple[str, str, str]] = [
    ("BTC",  "PF_XBTUSD",  "BTC-USDT-SWAP"),   # control (efficient)
    ("ETH",  "PF_ETHUSD",  "ETH-USDT-SWAP"),   # control (efficient)
    ("SOL",  "PF_SOLUSD",  "SOL-USDT-SWAP"),   # large alt
    ("XRP",  "PF_XRPUSD",  "XRP-USDT-SWAP"),
    ("DOGE", "PF_DOGEUSD", "DOGE-USDT-SWAP"),
    ("LTC",  "PF_LTCUSD",  "LTC-USDT-SWAP"),   # mid
    ("NEAR", "PF_NEARUSD", "NEAR-USDT-SWAP"),
    ("UNI",  "PF_UNIUSD",  "UNI-USDT-SWAP"),
    ("SUI",  "PF_SUIUSD",  "SUI-USDT-SWAP"),
    ("ARB",  "PF_ARBUSD",  "ARB-USDT-SWAP"),   # smaller
]

# Map a Kraken resolution string to OKX's bar-size spelling.
_OKX_BAR = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1H", "4h": "4H", "12h": "12H", "1d": "1D"}

OKX_SWAP_TICKERS_URL = "https://www.okx.com/api/v5/market/tickers"


# ============================================================
# Liquidity indicator
# ============================================================

def fetch_basket_liquidity(basket: list[tuple[str, str, str]] = None) -> dict[str, float]:
    """Return {base: approx 24h USD notional} from OKX swap tickers (one request).

    Uses `volCcy24h * last` (24h base-unit volume x last price) as a rough USD
    liquidity proxy. Best-effort: an asset missing a ticker gets 0.0.
    """
    basket = basket or DEFAULT_BASKET
    data = _http_get_json(OKX_SWAP_TICKERS_URL, params={"instType": "SWAP"})
    tickers = {t["instId"]: t for t in data.get("data", [])}
    out: dict[str, float] = {}
    for base, _k, okx_id in basket:
        t = tickers.get(okx_id, {})
        try:
            out[base] = float(t.get("volCcy24h") or 0.0) * float(t.get("last") or 0.0)
        except (TypeError, ValueError):
            out[base] = 0.0
    return out


# ============================================================
# Basket loader
# ============================================================

def load_cross_venue_basket(
    basket: list[tuple[str, str, str]] = None,
    resolution: str = "5m",
    start: str | None = None,
    end: str | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Fetch both venues' OHLCV for each basket asset over [start, end).

    Args:
        basket: list of (base, kraken_symbol, okx_inst_id); defaults to DEFAULT_BASKET.
        resolution: Kraken resolution (e.g. "5m"); mapped to OKX's bar spelling.
        start, end: ISO date strings (UTC).
        cache_dir, force_refresh: passed to both underlying loaders.

    Returns:
        {base: (kraken_df, okx_df)} for every asset that loaded from BOTH venues.
        An asset that fails on either venue is logged and skipped (so one bad
        symbol never sinks a multi-asset scan).
    """
    basket = basket or DEFAULT_BASKET
    if resolution not in _OKX_BAR:
        raise ValueError(f"Unsupported resolution {resolution!r}; known: {sorted(_OKX_BAR)}")
    okx_bar = _OKX_BAR[resolution]

    out: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for base, kraken_symbol, okx_inst_id in basket:
        try:
            kdf = fetch_ohlcv(kraken_symbol, start, end, resolution)
            time.sleep(PER_REQUEST_DELAY_SEC)
            odf = fetch_okx_ohlcv(okx_inst_id, bar=okx_bar, start=start, end=end,
                                  cache_dir=cache_dir, force_refresh=force_refresh)
            out[base] = (kdf, odf)
            logger.info("Loaded %s: kraken=%d bars, okx=%d bars", base, len(kdf), len(odf))
        except Exception as e:  # keep scanning the rest of the basket
            logger.warning("Skipping %s (load failed): %s", base, e)
        time.sleep(PER_REQUEST_DELAY_SEC)
    return out
