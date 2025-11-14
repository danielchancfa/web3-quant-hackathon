"""
Historical Data Pipeline
-----------------------
Fetches market, on-chain, and sentiment datasets and stores them in SQLite for model training.
"""

import sqlite3
import time
from pathlib import Path
from typing import List, Optional
import time
from datetime import datetime, timezone
import pandas as pd

from config import get_config, get_fear_greed_csv_path
from .data_engine import DataEngine
from .horus_api import (
    get_transaction_count,
    get_mining_work,
    get_chain_tvl,
    get_whale_net_flow,
    get_market_price,
)
from .coinmarketcap_api import (
    get_fear_and_greed_historical,
    get_ohlcv_historical,
    get_top_market_cap_symbols,
)
from .roostoo_api import get_exchange_info
from .ccxt_source import fetch_ohlcv_via_ccxt, DEFAULT_EXCHANGES as CCXT_DEFAULT_EXCHANGES

DEFAULT_PAIRS = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'BNB/USD', 'XRP/USD',
    'DOGE/USD', 'ADA/USD', 'MATIC/USD', 'LTC/USD'
]

HORUS_TRANSACTION_CHAINS = ['bitcoin']
HORUS_TVL_CHAINS = ['bitcoin', 'ethereum', 'solana', 'tron', 'bsc', 'base']
MINING_CHAINS = ['bitcoin']
TOP_MARKET_CAP_LIMIT = 20

HORUS_SUPPORTED_ASSETS = {
    "BTC", "ETH", "XRP", "BNB", "SOL", "DOGE", "TRX", "ADA", "XLM", "WBTC",
    "SUI", "HBAR", "LINK", "BCH", "WBETH", "UNI", "AVAX", "SHIB", "TON",
    "LTC", "DOT", "PEPE", "AAVE", "ONDO", "TAO", "WLD", "APT", "NEAR",
    "ARB", "ICP", "ETC", "FIL", "TRUMP", "OP", "ALGO", "POL", "BONK",
    "ENA", "ENS", "VET", "SEI", "RENDER", "FET", "ATOM", "VIRTUAL",
    "SKY", "BNSOL", "RAY", "TIA", "JTO", "JUP", "QNT", "FORM", "INJ", "STX"
}
HORUS_ASSETS = sorted({p.split('/')[0].upper() for p in DEFAULT_PAIRS} & HORUS_SUPPORTED_ASSETS)
HORUS_MARKET_INTERVALS = ['1h', '1d']
HORUS_INTERVAL_SECONDS = {
    '15m': 15 * 60 * 96 * 14,   # 2 weeks
    '1h': 3600 * 24 * 30,       # 30 days
    '1d': 86400 * 365 * 5,      # 5 years
}
HORUS_RETRY_DELAYS = (1, 2, 4, 8, 16)
HORUS_REQUEST_THROTTLE = 0.5  # seconds between successful requests
MAX_HORUS_PRICE_REQUESTS = 200
CMC_INTERVAL_MAP = {
    '1d': '1d',
    '1h': '1h',
}
CCXT_DEFAULT_INTERVALS = ['1h', '1d']


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


def _fetch_full_horus_price_series(asset: str, interval: str) -> pd.DataFrame:
    """Iteratively fetch the full price history for an asset/interval from Horus."""
    step_seconds = HORUS_INTERVAL_SECONDS.get(interval, 3600 * 24 * 30)
    end_ts = int(datetime.now(timezone.utc).timestamp()) + 60
    frames: List[pd.DataFrame] = []
    seen: set[int] = set()

    for _ in range(MAX_HORUS_PRICE_REQUESTS):
        start_ts = max(0, end_ts - step_seconds)
        df = None

        for delay in HORUS_RETRY_DELAYS:
            df = get_market_price(
                asset=asset,
                interval=interval,
                start=start_ts,
                end=end_ts,
                as_dataframe=True,
            )
            if df is not None:
                break
            time.sleep(delay)

        if df is None or df.empty:
            break

        df = df.drop_duplicates(subset=['timestamp'])
        df = df[df['timestamp'] < end_ts]
        df = df[~df['timestamp'].isin(seen)]
        if df.empty:
            break

        frames.append(df)
        seen.update(int(ts) for ts in df['timestamp'].tolist())

        new_end = int(df['timestamp'].min()) - 1
        if new_end <= 0 or new_end >= end_ts:
            break
        end_ts = new_end
        time.sleep(HORUS_REQUEST_THROTTLE)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['timestamp'])
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    return combined


def fetch_and_store_market_prices(
    engine: DataEngine,
    conn: sqlite3.Connection,
    assets: List[str],
    intervals: List[str]
) -> None:
    frames: List[pd.DataFrame] = []
    for asset in assets:
        for interval in intervals:
            df = _fetch_full_horus_price_series(asset, interval)
            if df is None or df.empty:
                print(f"[WARN] No Horus market price data for {asset} ({interval})")
                continue
            df = df.copy()
            df['asset'] = asset.upper()
            df['interval'] = interval
            df['timestamp'] = df['timestamp'].astype(int)
            frames.append(df)

    if not frames:
        print("[WARN] No Horus market price data fetched.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[['asset', 'interval', 'timestamp', 'datetime', 'price']]
    _store_dataframe(conn, combined, 'horus_market_price')
    store_horus_prices_as_ohlcv(engine, combined)


def store_horus_prices_as_ohlcv(engine: DataEngine, price_df: pd.DataFrame) -> None:
    grouped = price_df.groupby(['asset', 'interval'])
    for (asset, interval), group in grouped:
        if group.empty:
            continue
        pair = f"{asset}/USD"
        group_sorted = group.sort_values('datetime')
        dt = pd.to_datetime(group_sorted['datetime'], errors='coerce', utc=True)
        mask = ~dt.isna()
        dt = dt[mask]
        if dt.empty:
            continue
        if hasattr(dt, "dt"):
            dt = dt.dt.tz_convert(None)
        prices = group_sorted.loc[mask, 'price'].values
        ohlc = pd.DataFrame({
            'open': prices,
            'high': prices,
            'low': prices,
            'close': prices,
            'volume': 0.0,
        }, index=dt)
        try:
            engine.save_ohlcv_to_db(pair, ohlc, interval=interval)
        except Exception as exc:
            print(f"[WARN] Failed to store Horus price for {pair} ({interval}): {exc}")


def fetch_and_store_ccxt_ohlcv(
    engine: DataEngine,
    pairs: List[str],
    intervals: List[str],
    exchanges: Optional[List[str]] = None,
    limit: int = 1000,
    max_fetches: int = 20,
    since_ms: Optional[int] = None,
) -> None:
    exchanges = exchanges or list(CCXT_DEFAULT_EXCHANGES)
    print("\nFetching CCXT OHLCV series...")
    for interval in intervals:
        try:
            print(f"  Interval: {interval} (limit={limit})")
            for pair in pairs:
                try:
                    df = fetch_ohlcv_via_ccxt(
                        pair,
                        interval,
                        limit=limit,
                        exchanges=exchanges,
                        since=since_ms,
                        max_fetches=max_fetches,
                    )
                    if df.empty:
                        print(f"    [WARN] No CCXT data for {pair} ({interval})")
                        continue
                    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
                    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_convert(None)
                    df = df.set_index('datetime')
                    engine.save_ohlcv_to_db(pair, df, interval=interval)
                    exch = df.attrs.get('exchange') if hasattr(df, 'attrs') else None
                    print(f"    [OK] Stored {len(df)} rows for {pair} ({interval}) via CCXT{f' [{exch}]' if exch else ''}")
                except Exception as exc:
                    print(f"    [ERROR] Failed CCXT fetch for {pair} ({interval}): {exc}")
        except Exception as exc:
            print(f"  [ERROR] CCXT interval {interval} failed: {exc}")


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
            engine.update_market_data(DEFAULT_PAIRS, interval='1m', limit=300, incremental=False)
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

        if config.use_ccxt_ohlcv:
            ccxt_pairs = config.default_pairs or DEFAULT_PAIRS
            ccxt_exchanges = None
            if config.ccxt_exchanges:
                ccxt_exchanges = [ex.strip() for ex in config.ccxt_exchanges.split(',') if ex.strip()]
            ccxt_since_ms = None
            if config.ccxt_start_date:
                try:
                    dt = datetime.fromisoformat(config.ccxt_start_date)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    ccxt_since_ms = int(dt.timestamp() * 1000)
                except Exception:
                    print(f"[WARN] Could not parse CCXT_START_DATE '{config.ccxt_start_date}', ignoring.")
            fetch_and_store_ccxt_ohlcv(
                engine,
                ccxt_pairs,
                CCXT_DEFAULT_INTERVALS,
                exchanges=ccxt_exchanges,
                limit=config.ccxt_ohlcv_limit,
                max_fetches=config.ccxt_max_fetches,
                since_ms=ccxt_since_ms,
            )

        # Horus metrics
        print("\nFetching Horus metrics...")
        fetch_and_store_transaction_count(conn, HORUS_TRANSACTION_CHAINS)
        fetch_and_store_mining_work(conn, MINING_CHAINS)
        fetch_and_store_chain_tvl(conn, HORUS_TVL_CHAINS)
        fetch_and_store_whale_net_flow(conn, ['bitcoin'])
        if config.use_horus_market_price and HORUS_ASSETS:
            print("\nFetching Horus market prices (1h & 1d)...")
            fetch_and_store_market_prices(engine, conn, HORUS_ASSETS, HORUS_MARKET_INTERVALS)
        else:
            print("\nSkipping Horus market prices (disabled or no supported assets).")

        # CoinMarketCap sentiment
        print("\nFetching CoinMarketCap Fear & Greed index...")
        fetch_and_store_fear_greed(conn, csv_path if csv_path.exists() else None)

        print("\nAll datasets fetched and stored successfully.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
