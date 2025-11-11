"""
Historical Data Pipeline
-----------------------
Fetches market, on-chain, and sentiment datasets and stores them in SQLite for model training.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional
import time
import pandas as pd

from config import get_config, get_fear_greed_csv_path
from .data_engine import DataEngine
from .horus_api import (
    get_transaction_count,
    get_mining_work,
    get_chain_tvl,
    get_whale_net_flow,
)
from .coinmarketcap_api import (
    get_fear_and_greed_historical,
    get_ohlcv_historical,
    get_top_market_cap_symbols,
)
from .roostoo_api import get_exchange_info

DEFAULT_PAIRS = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'BNB/USD', 'XRP/USD',
    'DOGE/USD', 'ADA/USD', 'MATIC/USD', 'LTC/USD'
]

HORUS_TRANSACTION_CHAINS = ['bitcoin']
HORUS_TVL_CHAINS = ['bitcoin', 'ethereum', 'solana', 'tron', 'bsc', 'base']
MINING_CHAINS = ['bitcoin']
TOP_MARKET_CAP_LIMIT = 20
CMC_INTERVAL_MAP = {
    '1d': '1d',
    '1h': '1h',
}


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'datetime' in df.columns and pd.api.types.is_datetime64_any_dtype(df['datetime']):
        if pd.api.types.is_datetime64tz_dtype(df['datetime']):
            df['datetime'] = df['datetime'].dt.tz_convert('UTC').dt.tz_localize(None)
        df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    return df


def _store_dataframe(conn: sqlite3.Connection, df: pd.DataFrame, table_name: str) -> None:
    if df is None or df.empty:
        print(f"[SKIP] {table_name}: no data")
        return
    df = _prepare_dataframe(df)
    df = df.drop_duplicates()
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"[OK] Stored {len(df)} rows into '{table_name}'")


def fetch_and_store_transaction_count(conn: sqlite3.Connection, chains: List[str]) -> None:
    frames = []
    for chain in chains:
        df = get_transaction_count(chain=chain, interval='1d', as_dataframe=True)
        if df is not None and not df.empty:
            frames.append(df)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        _store_dataframe(conn, combined, 'horus_transaction_count')
    else:
        print("[WARN] No transaction count data fetched.")


def fetch_and_store_mining_work(conn: sqlite3.Connection, chains: List[str]) -> None:
    frames = []
    for chain in chains:
        df = get_mining_work(chain=chain, interval='1d', as_dataframe=True)
        if df is not None and not df.empty:
            frames.append(df)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        _store_dataframe(conn, combined, 'horus_mining_work')
    else:
        print("[WARN] No mining work data fetched.")


def fetch_and_store_chain_tvl(conn: sqlite3.Connection, chains: List[str]) -> None:
    df = get_chain_tvl(chains, interval='1d', as_dataframe=True)
    if df is not None and not df.empty:
        _store_dataframe(conn, df, 'horus_chain_tvl')
    else:
        print("[WARN] No chain TVL data fetched.")


def fetch_and_store_whale_net_flow(conn: sqlite3.Connection, chains: List[str]) -> None:
    df = get_whale_net_flow(chains, interval='1d', as_dataframe=True)
    if df is not None and not df.empty:
        _store_dataframe(conn, df, 'horus_whale_net_flow')
    else:
        print("[WARN] No whale net flow data fetched.")


def get_available_pairs() -> List[str]:
    info = get_exchange_info()
    if not info:
        return DEFAULT_PAIRS
    potential = info.get('TradePairs') or info.get('tradePairs')
    pairs: List[str] = []

    if isinstance(potential, dict):
        pairs = list(potential.keys())
    elif isinstance(potential, list):
        pairs = potential

    if not pairs:
        fallback = info.get('symbols') or info.get('pairs') or []
        for entry in fallback:
            if isinstance(entry, dict):
                pair = entry.get('symbol') or entry.get('pair')
            else:
                pair = entry
            if pair:
                pairs.append(pair)

    normalized: List[str] = []
    for pair in pairs:
        if not isinstance(pair, str):
            continue
        pair = pair.replace('-', '/')
        if '/' not in pair and len(pair) > 3:
            # Attempt to insert slash before quote (assume last 3 chars are quote)
            pair = f"{pair[:-3]}/{pair[-3:]}"
        normalized.append(pair.upper())

    if not normalized:
        normalized = DEFAULT_PAIRS

    top_symbols = get_top_market_cap_symbols(limit=TOP_MARKET_CAP_LIMIT) or []
    if top_symbols:
        filtered = [pair for pair in normalized if pair.split('/')[0] in top_symbols]
        if filtered:
            normalized = filtered

    return normalized[:TOP_MARKET_CAP_LIMIT]


def fetch_and_store_cmc_ohlcv(
    engine: DataEngine,
    pairs: List[str],
    interval: str,
    count: int = 1000,
    sleep_seconds: float = 0.6
) -> None:
    cmc_interval = CMC_INTERVAL_MAP.get(interval)
    if cmc_interval is None:
        raise ValueError(f"Unsupported interval '{interval}' for CoinMarketCap OHLCV fetch.")

    print(f"\nFetching CoinMarketCap {cmc_interval} OHLCV for {len(pairs)} pairs...")
    for pair in pairs:
        base = pair.split('/')[0].upper().strip()
        try:
            df = get_ohlcv_historical(symbol=base, interval=cmc_interval, count=count, as_dataframe=True)
            if df is None or df.empty:
                print(f"  [WARN] No OHLCV data for {pair} ({base})")
                continue

            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['datetime'] = df['datetime'].dt.tz_localize(None)
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            if df.empty:
                print(f"  [WARN] No valid OHLCV rows for {pair}")
                continue

            df = df.set_index('datetime')[['open', 'high', 'low', 'close', 'volume']]
            engine.save_ohlcv_to_db(pair, df, interval=interval)
            print(f"  [OK] Stored {len(df)} {interval} rows for {pair}")
        except Exception as exc:
            print(f"  [ERROR] Failed to fetch {pair} ({base}): {exc}")
        time.sleep(sleep_seconds)


def fetch_and_store_fear_greed(conn: sqlite3.Connection, csv_path: Optional[Path]) -> None:
    df = None

    if csv_path and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'], utc=True)
                df['timestamp'] = df['datetime'].view('int64') // 10**9
                df = df.rename(columns={
                    'fng_value': 'fear_greed_value',
                    'fng_classification': 'fear_greed_class'
                })
                df['interval'] = '1d'
                df = df[['datetime', 'timestamp', 'interval', 'fear_greed_value', 'fear_greed_class']]
                print(f"[OK] Loaded Fear & Greed data from CSV ({csv_path})")
            else:
                print(f"[WARN] CSV {csv_path} missing 'date' column. Ignoring file.")
                df = None
        except Exception as exc:
            print(f"[WARN] Failed to load Fear & Greed CSV {csv_path}: {exc}")
            df = None

    if df is None:
        df = get_fear_and_greed_historical(as_dataframe=True)
        if df is not None and not df.empty:
            print("[OK] Loaded Fear & Greed data from CoinMarketCap API")

    if df is not None and not df.empty:
        _store_dataframe(conn, df, 'coinmarketcap_fear_greed')
    else:
        print("[WARN] No fear & greed data fetched.")


def main() -> None:
    config = get_config()
    db_path = Path(config.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using SQLite database at: {db_path}")
    conn = sqlite3.connect(db_path)

    try:
        fear_greed_csv = get_fear_greed_csv_path()
        csv_path = Path(fear_greed_csv) if fear_greed_csv else Path(__file__).parent / 'CryptoGreedFear.csv'
        if csv_path.exists():
            print(f"Fear & Greed CSV detected at: {csv_path}")
        else:
            print(f"Fear & Greed CSV not found at {csv_path}, falling back to API.")

        # Market data
        engine = DataEngine()
        if config.fetch_intraday_market_data:
            print("\nFetching market data...")
            engine.update_market_data(DEFAULT_PAIRS, interval='1m', limit=1000, incremental=False)
            stats = engine.get_database_stats()
            print(f"[OK] Market data update complete. Total OHLCV rows: {stats.get('total_rows', 'N/A')}")
        else:
            print("\nSkipping intraday market data fetch (fetch_intraday_market_data=false).")

        if config.use_roostoo_intraday:
            print("\nSkipping CoinMarketCap OHLCV fetch (using Roostoo intraday data).")
        else:
            print("\nFetching CoinMarketCap OHLCV series...")
            cmc_pairs = get_available_pairs()
            fetch_and_store_cmc_ohlcv(engine, cmc_pairs, interval='1d', count=700)
            fetch_and_store_cmc_ohlcv(engine, cmc_pairs, interval='1h', count=720)

        # Horus metrics
        print("\nFetching Horus metrics...")
        fetch_and_store_transaction_count(conn, HORUS_TRANSACTION_CHAINS)
        fetch_and_store_mining_work(conn, MINING_CHAINS)
        fetch_and_store_chain_tvl(conn, HORUS_TVL_CHAINS)
        fetch_and_store_whale_net_flow(conn, ['bitcoin'])

        # CoinMarketCap sentiment
        print("\nFetching CoinMarketCap Fear & Greed index...")
        fetch_and_store_fear_greed(conn, csv_path if csv_path.exists() else None)

        print("\nAll datasets fetched and stored successfully.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
