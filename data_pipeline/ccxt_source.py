"""
CCXT-based OHLCV data fetcher.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_EXCHANGES: Sequence[str] = (
    "binance",
    "okx",
    "kucoin",
    "kraken",
    "bitfinex",
)

TIMEFRAME_MAP = {
    "1h": "1h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1d",
}


@lru_cache(maxsize=16)
def _load_exchange(exchange_id: str) -> ccxt.Exchange:
    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})
    exchange.load_markets()
    return exchange


def _candidate_symbols(pair: str) -> List[str]:
    pair = pair.upper().strip()
    candidates = [pair]
    if "/" in pair:
        base, quote = pair.split("/")
        quote = quote.upper()
        bases = {base}
        quotes = {quote}
        if quote == "USD":
            quotes.update({"USDT", "USDC"})
        elif quote == "USDT":
            quotes.add("USD")
            quotes.add("USDC")
        for q in quotes:
            candidates.append(f"{base}/{q}")
        if quote.endswith("USD") and quote != "USD":
            candidates.append(f"{base}/USD")
    else:
        # e.g. BTCUSD -> BTC/USD, BTC/USDT
        if pair.endswith("USD"):
            base = pair[:-3]
            candidates.extend([f"{base}/USD", f"{base}/USDT", f"{base}/USDC"])
    return list(dict.fromkeys(candidates))


def _resolve_symbol(exchange: ccxt.Exchange, pair: str) -> Optional[str]:
    candidates = _candidate_symbols(pair)
    markets = exchange.symbols
    for symbol in candidates:
        if symbol in markets:
            return symbol
    return None


def fetch_ohlcv_via_ccxt(
    pair: str,
    interval: str,
    limit: int = 1000,
    exchanges: Iterable[str] = DEFAULT_EXCHANGES,
    since: Optional[int] = None,
    max_fetches: int = 20,
) -> pd.DataFrame:
    timeframe = TIMEFRAME_MAP.get(interval)
    if timeframe is None:
        raise ValueError(f"Unsupported interval '{interval}' for CCXT fetch.")

    for exchange_id in exchanges:
        try:
            exchange = _load_exchange(exchange_id)
            symbol = _resolve_symbol(exchange, pair)
            if symbol is None:
                logger.debug(f"{exchange_id}: No symbol for {pair}")
                continue

            logger.info(f"Fetching {interval} OHLCV for {pair} ({symbol}) via {exchange_id}")
            all_rows: List[List[float]] = []
            fetch_since = since
            timeframe_ms = exchange.parse_timeframe(timeframe) * 1000

            for _ in range(max_fetches):
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit)
                if not ohlcv:
                    break
                all_rows.extend(ohlcv)
                if len(ohlcv) < limit:
                    break
                last_ts = ohlcv[-1][0]
                fetch_since = last_ts + timeframe_ms

            if not all_rows:
                logger.warning(f"{exchange_id}: No OHLCV returned for {pair}")
                continue

            df = pd.DataFrame(
                all_rows,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            ).drop_duplicates(subset=["timestamp"])
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.dropna(subset=["open", "high", "low", "close"])
            if df.empty:
                continue
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["exchange"] = exchange_id
            df["symbol"] = symbol
            df.attrs["exchange"] = exchange_id
            df.attrs["symbol"] = symbol
            return df
        except Exception as exc:
            logger.warning(f"{exchange_id}: failed to fetch {pair} ({interval}): {exc}")
            continue

    return pd.DataFrame()

