"""
Data Engine - Data Layer for AI Web3 Trading Bot
-------------------------------------------------
Continuously collects, preprocesses, and feeds structured data to the Signal Engine.

Functions:
- Market Data: Fetch 1-min OHLCV and volume from Roostoo API
- Feature Engineering: Create technical indicators (returns, moving averages, volatility)
- Data Caching: Store and batch data for training and inference
- Data Normalization: Normalize and resample data for model input
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import sqlite3
import json
from contextlib import contextmanager

from .roostoo_api import get_klines, get_ticker, get_exchange_info
from .data_sources import MultiSourceDataFetcher
from .horus_api import (
    get_transaction_count,
    get_mining_work,
    get_chain_tvl,
    get_whale_net_flow,
    set_horus_api_key,
    get_horus_api_key
)
from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataEngine:
    """
    Data Engine for collecting and preprocessing market data.
    Uses SQLite database for efficient data storage and querying.
    """
    
    def __init__(self, db_path: Optional[str] = None, update_interval: Optional[int] = None, 
                 use_fallback: Optional[bool] = None, horus_api_key: Optional[str] = None):
        """
        Initialize the Data Engine.
        
        Args:
            db_path: Path to SQLite database file (default: from config)
            update_interval: Interval in seconds between data updates (default: from config)
            use_fallback: Whether to use fallback data sources (default: from config)
            horus_api_key: Horus API key for on-chain data (optional, overrides config)
        """
        # Load configuration
        config = get_config()
        
        # Use provided values or fall back to config
        self.db_path = Path(db_path or config.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.update_interval = update_interval if update_interval is not None else config.update_interval
        self.use_fallback = use_fallback if use_fallback is not None else config.use_fallback
        
        # In-memory cache for recently accessed data: {pair: DataFrame}
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.cache_max_size = config.cache_max_size
        
        # Initialize multi-source data fetcher
        self.data_fetcher = MultiSourceDataFetcher(
            primary_source='roostoo',
            fallback_sources=['binance'] if self.use_fallback else []
        )
        
        # Set Horus API key if provided (runtime override)
        if horus_api_key:
            set_horus_api_key(horus_api_key)
            logger.info("Horus API key set via parameter (runtime override)")
        elif get_horus_api_key():
            logger.info("Horus API key found in config")
        else:
            logger.warning("Horus API key not set. On-chain data features will be limited.")
        
        # Initialize database
        self._init_database()
        
        logger.info(f"DataEngine initialized with db_path={self.db_path}, update_interval={self.update_interval}s, use_fallback={self.use_fallback}")
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # OHLCV data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    datetime TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(pair, interval, timestamp)
                )
            ''')
            
            # Create indexes for efficient querying
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_pair_interval_timestamp 
                ON ohlcv(pair, interval, timestamp)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_pair_interval 
                ON ohlcv(pair, interval)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON ohlcv(timestamp)
            ''')
            
            # Metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Technical indicators table (optional - can store computed indicators)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    indicator_name TEXT NOT NULL,
                    indicator_value REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(pair, interval, timestamp, indicator_name),
                    FOREIGN KEY (pair, interval, timestamp) 
                    REFERENCES ohlcv(pair, interval, timestamp)
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_indicators_lookup 
                ON indicators(pair, interval, timestamp, indicator_name)
            ''')
            
            # On-chain data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS onchain_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_symbol TEXT NOT NULL,
                    network TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    datetime TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metric_value_str TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(token_symbol, network, timestamp, metric_name)
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_onchain_lookup 
                ON onchain_data(token_symbol, network, timestamp, metric_name)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_onchain_timestamp 
                ON onchain_data(timestamp)
            ''')
            
            # Exchange flows table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS exchange_flows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exchange TEXT NOT NULL,
                    network TEXT NOT NULL,
                    token_symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    datetime TEXT NOT NULL,
                    inflow REAL,
                    outflow REAL,
                    net_flow REAL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(exchange, network, token_symbol, timestamp)
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_exchange_flows_lookup 
                ON exchange_flows(exchange, network, token_symbol, timestamp)
            ''')
            
            logger.info("Database initialized successfully")
    
    def get_available_pairs(self) -> List[str]:
        """
        Get list of available trading pairs from Roostoo.
        
        Returns:
            List of trading pair strings (e.g., ['BTC/USD', 'ETH/USD', ...])
        """
        try:
            info = get_exchange_info()
            if info and 'TradePairs' in info:
                pairs = list(info['TradePairs'].keys())
                logger.info(f"Found {len(pairs)} available trading pairs")
                return pairs
            else:
                logger.warning("Could not retrieve trading pairs from exchange info")
                return []
        except Exception as e:
            logger.error(f"Error getting available pairs: {e}")
            return []
    
    def fetch_klines(
        self, 
        pair: str, 
        interval: str = '1m', 
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch kline/OHLCV data from primary or fallback data sources.
        
        Tries Roostoo first, falls back to Binance if Roostoo fails or returns 429 error.
        
        Args:
            pair: Trading pair (e.g., 'BTC/USD')
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of klines to fetch
            start_time: Start timestamp in milliseconds (optional)
            end_time: End timestamp in milliseconds (optional)
        
        Returns:
            DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            Returns None if all sources fail
        """
        try:
            # Use multi-source fetcher with automatic fallback
            df = self.data_fetcher.fetch_klines(
                pair=pair,
                interval=interval,
                limit=limit,
                start_time=start_time,
                end_time=end_time,
                use_roostoo=True,
                use_fallback=self.use_fallback
            )
            
            if df is not None and not df.empty:
                logger.info(f"Successfully fetched {len(df)} klines for {pair} ({interval})")
                return df
            else:
                logger.warning(f"Failed to fetch klines for {pair} from all sources")
                return None
            
        except Exception as e:
            logger.error(f"Error fetching klines for {pair}: {e}")
            return None
    
    def save_ohlcv_to_db(self, pair: str, df: pd.DataFrame, interval: str = '1m'):
        """
        Save OHLCV data to SQLite database.
        
        Args:
            pair: Trading pair
            df: DataFrame with OHLCV data (must have timestamp, open, high, low, close, volume)
            interval: Data interval
        """
        if df.empty:
            return
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare data for insertion
                data_to_insert = []
                for idx, row in df.iterrows():
                    # Handle both datetime index and timestamp column
                    if isinstance(idx, pd.Timestamp):
                        dt_str = idx.isoformat()
                        ts = int(idx.timestamp() * 1000)
                    else:
                        dt_str = pd.to_datetime(row['timestamp'], unit='ms').isoformat()
                        ts = int(row['timestamp'])
                    
                    data_to_insert.append((
                        pair,
                        interval,
                        ts,
                        dt_str,
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['volume'])
                    ))
                
                # Insert or replace (handles duplicates via UNIQUE constraint)
                cursor.executemany('''
                    INSERT OR REPLACE INTO ohlcv 
                    (pair, interval, timestamp, datetime, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', data_to_insert)
                
                # Update metadata
                metadata_key = f"{pair}_{interval}"
                if 'timestamp' in df.columns:
                    last_ts = int(df['timestamp'].iloc[-1])
                elif isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
                    last_ts = int(df.index[-1].timestamp() * 1000)
                else:
                    last_ts = None

                metadata_value = json.dumps({
                    'last_update': datetime.now().isoformat(),
                    'rows': len(df),
                    'last_timestamp': last_ts
                })
                cursor.execute('''
                    INSERT OR REPLACE INTO metadata (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (metadata_key, metadata_value))
                
                logger.debug(f"Saved {len(data_to_insert)} rows to database for {pair} ({interval})")
                
        except Exception as e:
            logger.error(f"Failed to save data to database for {pair}: {e}")
            raise
    
    def load_ohlcv_from_db(
        self, 
        pair: str, 
        interval: str = '1m',
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data from SQLite database.
        
        Args:
            pair: Trading pair
            interval: Data interval
            start_time: Start timestamp in milliseconds (optional)
            end_time: End timestamp in milliseconds (optional)
            limit: Maximum number of rows to return (optional, returns most recent)
        
        Returns:
            DataFrame with OHLCV data, or None if no data found
        """
        try:
            with self._get_connection() as conn:
                # Build query
                query = '''
                    SELECT timestamp, datetime, open, high, low, close, volume
                    FROM ohlcv
                    WHERE pair = ? AND interval = ?
                '''
                params = [pair, interval]
                
                if start_time:
                    query += ' AND timestamp >= ?'
                    params.append(start_time)
                
                if end_time:
                    query += ' AND timestamp <= ?'
                    params.append(end_time)
                
                if limit:
                    # For limit, get most recent data
                    query += ' ORDER BY timestamp DESC LIMIT ?'
                    params.append(limit)
                    df = pd.read_sql_query(query, conn, params=params)
                    if not df.empty:
                        # Reverse to chronological order
                        df = df.sort_values('timestamp').reset_index(drop=True)
                else:
                    query += ' ORDER BY timestamp'
                    df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    return None
                
                # Convert to proper format
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime').sort_index()
                
                logger.debug(f"Loaded {len(df)} rows from database for {pair} ({interval})")
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
        except Exception as e:
            logger.error(f"Failed to load data from database for {pair}: {e}")
            return None
    
    def get_latest_timestamp(self, pair: str, interval: str = '1m') -> Optional[int]:
        """
        Get the latest timestamp in database for a pair/interval.
        
        Args:
            pair: Trading pair
            interval: Data interval
        
        Returns:
            Latest timestamp in milliseconds, or None if no data
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT MAX(timestamp) as latest
                    FROM ohlcv
                    WHERE pair = ? AND interval = ?
                ''', (pair, interval))
                result = cursor.fetchone()
                return result['latest'] if result and result['latest'] else None
        except Exception as e:
            logger.error(f"Failed to get latest timestamp for {pair}: {e}")
            return None
    
    def update_market_data(
        self, 
        pairs: List[str], 
        interval: str = '1m',
        limit: int = 1000,
        use_cache: bool = True,
        incremental: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Update market data for specified trading pairs.
        
        Args:
            pairs: List of trading pairs to update
            interval: Kline interval
            limit: Number of klines to fetch
            use_cache: Whether to load existing data from database
            incremental: If True, only fetch data newer than latest in database
        
        Returns:
            Dictionary of {pair: DataFrame} with updated data
        """
        updated_data = {}
        
        for pair in pairs:
            try:
                # Get latest timestamp from database if incremental update
                latest_ts = None
                if use_cache and incremental:
                    latest_ts = self.get_latest_timestamp(pair, interval)
                
                # Fetch new data from API
                new_df = self.fetch_klines(pair, interval=interval, limit=limit)
                
                if new_df is None:
                    # If fetch failed, try to load from database
                    if use_cache:
                        db_df = self.load_ohlcv_from_db(pair, interval, limit=limit)
                        if db_df is not None:
                            logger.info(f"Using database data for {pair} (API fetch failed)")
                            updated_data[pair] = db_df
                            # Update in-memory cache
                            self._update_memory_cache(pair, db_df)
                    continue
                
                # Filter out data we already have (if incremental)
                if latest_ts and len(new_df) > 0:
                    new_df = new_df[new_df['timestamp'] > latest_ts]
                    if new_df.empty:
                        logger.debug(f"No new data for {pair}, using existing data from database")
                        db_df = self.load_ohlcv_from_db(pair, interval, limit=limit)
                        if db_df is not None:
                            updated_data[pair] = db_df
                            self._update_memory_cache(pair, db_df)
                        continue
                
                # Save new data to database
                if not new_df.empty:
                    self.save_ohlcv_to_db(pair, new_df, interval)
                    logger.info(f"Saved {len(new_df)} new rows to database for {pair}")
                
                # Load complete dataset from database (merges automatically)
                if use_cache:
                    complete_df = self.load_ohlcv_from_db(pair, interval, limit=None)
                    if complete_df is not None:
                        updated_data[pair] = complete_df
                        self._update_memory_cache(pair, complete_df)
                        logger.info(f"Updated {pair}: {len(complete_df)} total rows in database")
                    else:
                        # Fallback to newly fetched data if database load fails
                        updated_data[pair] = new_df
                        self._update_memory_cache(pair, new_df)
                else:
                    updated_data[pair] = new_df
                    self._update_memory_cache(pair, new_df)
                
            except Exception as e:
                logger.error(f"Error updating data for {pair}: {e}")
                # Try to load from database as fallback
                if use_cache:
                    db_df = self.load_ohlcv_from_db(pair, interval, limit=limit)
                    if db_df is not None:
                        updated_data[pair] = db_df
                        self._update_memory_cache(pair, db_df)
        
        return updated_data
    
    def _update_memory_cache(self, pair: str, df: pd.DataFrame):
        """
        Update in-memory cache, keeping only the most recently accessed pairs.
        
        Args:
            pair: Trading pair
            df: DataFrame to cache
        """
        # Remove oldest entry if cache is full
        if len(self.market_data) >= self.cache_max_size and pair not in self.market_data:
            # Remove least recently used (simple: remove first)
            oldest_pair = next(iter(self.market_data))
            del self.market_data[oldest_pair]
            logger.debug(f"Removed {oldest_pair} from memory cache (max size: {self.cache_max_size})")
        
        self.market_data[pair] = df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added technical indicators
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volatility_annualized'] = df['volatility'] * np.sqrt(252 * 24 * 60)  # Annualized for 1-min data
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
        bb_std_val = df['close'].rolling(window=bb_window).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price position (relative to high-low range)
        df['price_position'] = (df['close'] - df['low'].rolling(window=20).min()) / (
            df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
        )
        
        logger.debug(f"Added technical indicators to DataFrame with {len(df)} rows")
        return df
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize data for model input.
        
        Args:
            df: DataFrame to normalize
            method: Normalization method ('minmax', 'zscore', 'robust')
        
        Returns:
            Normalized DataFrame
        """
        df = df.copy()
        
        # Select numeric columns to normalize (exclude timestamp)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'timestamp' in numeric_cols:
            numeric_cols.remove('timestamp')
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            for col in numeric_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
        elif method == 'robust':
            for col in numeric_cols:
                median_val = df[col].median()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                if iqr > 0:
                    df[col] = (df[col] - median_val) / iqr
        
        return df
    
    def resample_data(self, df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """
        Resample data to a different time interval.
        
        Args:
            df: DataFrame with datetime index
            target_interval: Target interval (e.g., '5m', '1h', '1d')
        
        Returns:
            Resampled DataFrame
        """
        if df.empty:
            return df
        
        # Resample OHLCV data
        resampled = pd.DataFrame()
        resampled['open'] = df['open'].resample(target_interval).first()
        resampled['high'] = df['high'].resample(target_interval).max()
        resampled['low'] = df['low'].resample(target_interval).min()
        resampled['close'] = df['close'].resample(target_interval).last()
        resampled['volume'] = df['volume'].resample(target_interval).sum()
        resampled['timestamp'] = resampled.index.astype(np.int64) // 10**6  # Convert to milliseconds
        
        # Drop rows with NaN (incomplete intervals)
        resampled = resampled.dropna()
        
        return resampled
    
    def get_latest_features(
        self, 
        pair: str, 
        interval: str = '1m',
        lookback: int = 100,
        include_indicators: bool = True,
        normalize: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Get latest features for a trading pair, ready for model input.
        Loads from database if not in memory cache.
        
        Args:
            pair: Trading pair
            interval: Data interval
            lookback: Number of recent rows to return
            include_indicators: Whether to include technical indicators
            normalize: Whether to normalize the data
        
        Returns:
            DataFrame with features, or None if data not available
        """
        # Try memory cache first
        if pair in self.market_data:
            df = self.market_data[pair].copy()
        else:
            # Load from database
            df = self.load_ohlcv_from_db(pair, interval, limit=lookback * 2)  # Load extra for indicators
            if df is None:
                logger.warning(f"No data available for {pair} in database or cache")
                return None
            # Update memory cache
            self._update_memory_cache(pair, df)
            df = df.copy()
        
        if df.empty:
            return None
        
        # Get latest data (need extra rows for indicator calculations)
        if include_indicators:
            # Need more rows for indicators (e.g., 200-period MA needs 200+ rows)
            indicator_window = 200
            df = df.tail(max(lookback + indicator_window, indicator_window * 2))
        else:
            df = df.tail(lookback)
        
        # Add technical indicators
        if include_indicators:
            df = self.add_technical_indicators(df)
            # Drop rows with NaN from indicators
            df = df.dropna()
            # Get only the requested lookback after indicators
            df = df.tail(lookback)
        
        # Normalize if requested
        if normalize:
            df = self.normalize_data(df)
        
        return df
    
    def query_data(
        self,
        pair: str,
        interval: str = '1m',
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Query OHLCV data from database with flexible filtering.
        
        Args:
            pair: Trading pair
            interval: Data interval
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Maximum number of rows
        
        Returns:
            DataFrame with OHLCV data
        """
        return self.load_ohlcv_from_db(pair, interval, start_time, end_time, limit)
    
    def get_database_stats(self) -> Dict:
        """
        Get statistics about data in the database.
        
        Returns:
            Dictionary with database statistics
        """
        stats = {}
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Total rows
                cursor.execute('SELECT COUNT(*) as total FROM ohlcv')
                stats['total_rows'] = cursor.fetchone()['total']
                
                # Pairs
                cursor.execute('SELECT DISTINCT pair FROM ohlcv')
                stats['pairs'] = [row['pair'] for row in cursor.fetchall()]
                
                # Intervals
                cursor.execute('SELECT DISTINCT interval FROM ohlcv')
                stats['intervals'] = [row['interval'] for row in cursor.fetchall()]
                
                # Data range
                cursor.execute('SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM ohlcv')
                result = cursor.fetchone()
                if result and result['min_ts']:
                    stats['date_range'] = {
                        'start': datetime.fromtimestamp(result['min_ts'] / 1000),
                        'end': datetime.fromtimestamp(result['max_ts'] / 1000)
                    }
                
                # Rows per pair
                cursor.execute('''
                    SELECT pair, interval, COUNT(*) as count 
                    FROM ohlcv 
                    GROUP BY pair, interval
                    ORDER BY count DESC
                ''')
                stats['rows_per_pair'] = [
                    {'pair': row['pair'], 'interval': row['interval'], 'count': row['count']}
                    for row in cursor.fetchall()
                ]
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
        
        return stats
    
    def fetch_onchain_data(
        self,
        token_symbol: str,
        network: str = 'ethereum',
        metrics: List[str] = None
    ) -> Optional[Dict]:
        """
        Fetch on-chain data for a token from Horus API.
        
        Args:
            token_symbol: Token symbol (e.g., 'ETH', 'BTC', 'USDT')
            network: Blockchain network (default: 'ethereum')
            metrics: List of metrics to fetch (default: all available)
        
        Returns:
            Dict with on-chain data
        """
        try:
            metrics = metrics or ['transaction_count', 'chain_tvl', 'whale_net_flow', 'mining_work']

            tx_df = None
            tvl_df = None
            whale_df = None
            mining_df = None

            if 'transaction_count' in metrics:
                tx_df = get_transaction_count(chain=network, interval='1d', as_dataframe=True)
            if 'chain_tvl' in metrics:
                tvl_df = get_chain_tvl(network, interval='1d', as_dataframe=True)
            if 'whale_net_flow' in metrics:
                whale_df = get_whale_net_flow(network, interval='1d', as_dataframe=True)
            if 'mining_work' in metrics and network == 'bitcoin':
                mining_df = get_mining_work(chain='bitcoin', interval='1d', as_dataframe=True)

            def _latest_value(df: Optional[pd.DataFrame], value_col: str) -> Optional[float]:
                if df is None or df.empty or value_col not in df.columns:
                    return None
                return pd.to_numeric(df[value_col].iloc[-1], errors='coerce')

            summary_metrics = {
                'transaction_count_latest': _latest_value(tx_df, 'transaction_count'),
                'chain_tvl_latest': _latest_value(tvl_df, 'tvl'),
                'whale_net_flow_latest': _latest_value(whale_df, 'whale_net_flow')
            }
            if mining_df is not None:
                summary_metrics['mining_work_latest'] = _latest_value(mining_df, 'mining_work')

            onchain_data = {
                'token_symbol': token_symbol,
                'network': network,
                'transaction_count_timeseries': tx_df,
                'chain_tvl_timeseries': tvl_df,
                'whale_net_flow_timeseries': whale_df,
                'mining_work_timeseries': mining_df,
                'summary_metrics': summary_metrics,
                'timestamp': int(time.time()),
                'datetime': datetime.now().isoformat()
            }
            
            logger.info(f"Fetched on-chain data for {token_symbol} on {network}")
            return onchain_data
            
        except Exception as e:
            logger.error(f"Error fetching on-chain data for {token_symbol}: {e}")
            return None
    
    def save_onchain_data_to_db(self, onchain_data: Dict):
        """
        Save on-chain data to SQLite database.
        
        Args:
            onchain_data: Dict with on-chain data from fetch_onchain_data()
        """
        if not onchain_data:
            return
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                token_symbol = onchain_data.get('token_symbol')
                network = onchain_data.get('network')
                timestamp = onchain_data.get('timestamp', int(time.time()))
                dt_str = onchain_data.get('datetime', datetime.now().isoformat())

                summary_metrics = onchain_data.get('summary_metrics', {})
                for metric_name, metric_value in summary_metrics.items():
                    if metric_value is not None:
                        cursor.execute('''
                            INSERT OR REPLACE INTO onchain_data 
                            (token_symbol, network, timestamp, datetime, metric_name, metric_value)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (token_symbol, network, timestamp * 1000, dt_str, metric_name, float(metric_value)))

                logger.debug(f"Saved on-chain summary metrics to database for {token_symbol}")

        except Exception as e:
            logger.error(f"Failed to save on-chain data to database: {e}")
    
    def fetch_exchange_flows(
        self,
        exchange: str,
        token_symbol: str,
        network: str = 'ethereum',
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch exchange inflow/outflow data and save to database.
        
        Args:
            exchange: Exchange name (e.g., 'binance', 'coinbase')
            token_symbol: Token symbol
            network: Blockchain network
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
        
        Returns:
            DataFrame with exchange flow data
        """
        try:
            # flows = get_exchange_flows(exchange, network, start_time, end_time) # This function is no longer imported
            
            # if not flows: # This function is no longer imported
            #     return None # This function is no longer imported
            
            # # Parse and save to database # This function is no longer imported
            # with self._get_connection() as conn: # This function is no longer imported
            #     cursor = conn.cursor() # This function is no longer imported
                
            #     # Parse flow data (format may vary) # This function is no longer imported
            #     if isinstance(flows, dict) and 'data' in flows: # This function is no longer imported
            #         data = flows['data'] # This function is no longer imported
            #         if isinstance(data, list): # This function is no longer imported
            #             for flow in data: # This function is no longer imported
            #                 timestamp = flow.get('timestamp') or flow.get('time') # This function is no longer imported
            #                 if isinstance(timestamp, str): # This function is no longer imported
            #                     timestamp = int(datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp()) # This function is no longer imported
                            
            #                 inflow = flow.get('inflow') or flow.get('in') # This function is no longer imported
            #                 outflow = flow.get('outflow') or flow.get('out') # This function is no longer imported
            #                 net_flow = flow.get('net_flow') or flow.get('net') # This function is no longer imported
                            
            #                 if net_flow is None and inflow is not None and outflow is not None: # This function is no longer imported
            #                     net_flow = inflow - outflow # This function is no longer imported
                            
            #                 if timestamp: # This function is no longer imported
            #                     dt_str = datetime.fromtimestamp(timestamp).isoformat() # This function is no longer imported
            #                     cursor.execute(''' # This function is no longer imported
            #                         INSERT OR REPLACE INTO exchange_flows  # This function is no longer imported
            #                         (exchange, network, token_symbol, timestamp, datetime, inflow, outflow, net_flow) # This function is no longer imported
            #                         VALUES (?, ?, ?, ?, ?, ?, ?, ?) # This function is no longer imported
            #                     ''', (exchange, network, token_symbol, timestamp * 1000, dt_str,  # This function is no longer imported
            #                          float(inflow) if inflow else None, # This function is no longer imported
            #                          float(outflow) if outflow else None, # This function is no longer imported
            #                          float(net_flow) if net_flow else None)) # This function is no longer imported
            
            # # Load from database as DataFrame # This function is no longer imported
            # return self.load_exchange_flows_from_db(exchange, token_symbol, network, start_time, end_time) # This function is no longer imported
            
            # Placeholder for actual exchange flow fetching logic
            logger.warning(f"Exchange flow fetching is not yet implemented for {exchange} on {network}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching exchange flows: {e}")
            return None
    
    def load_exchange_flows_from_db(
        self,
        exchange: str,
        token_symbol: str,
        network: str = 'ethereum',
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load exchange flows from database.
        
        Args:
            exchange: Exchange name
            token_symbol: Token symbol
            network: Blockchain network
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Maximum number of rows
        
        Returns:
            DataFrame with exchange flow data
        """
        try:
            with self._get_connection() as conn:
                query = '''
                    SELECT timestamp, datetime, inflow, outflow, net_flow
                    FROM exchange_flows
                    WHERE exchange = ? AND network = ? AND token_symbol = ?
                '''
                params = [exchange, network, token_symbol]
                
                if start_time:
                    query += ' AND timestamp >= ?'
                    params.append(start_time)
                
                if end_time:
                    query += ' AND timestamp <= ?'
                    params.append(end_time)
                
                if limit:
                    query += ' ORDER BY timestamp DESC LIMIT ?'
                    params.append(limit)
                    df = pd.read_sql_query(query, conn, params=params)
                    if not df.empty:
                        df = df.sort_values('timestamp').reset_index(drop=True)
                else:
                    query += ' ORDER BY timestamp'
                    df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    return None
                
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime').sort_index()
                
                return df
                
        except Exception as e:
            logger.error(f"Error loading exchange flows from database: {e}")
            return None
    
    def load_onchain_data_from_db(
        self,
        token_symbol: str,
        network: str = 'ethereum',
        metric_names: List[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load on-chain data from database.
        
        Args:
            token_symbol: Token symbol
            network: Blockchain network
            metric_names: List of metric names to load (default: all)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
        
        Returns:
            DataFrame with on-chain metrics (pivoted by metric_name)
        """
        try:
            with self._get_connection() as conn:
                query = '''
                    SELECT timestamp, datetime, metric_name, metric_value
                    FROM onchain_data
                    WHERE token_symbol = ? AND network = ?
                '''
                params = [token_symbol, network]
                
                if metric_names:
                    placeholders = ','.join(['?'] * len(metric_names))
                    query += f' AND metric_name IN ({placeholders})'
                    params.extend(metric_names)
                
                if start_time:
                    query += ' AND timestamp >= ?'
                    params.append(start_time)
                
                if end_time:
                    query += ' AND timestamp <= ?'
                    params.append(end_time)
                
                query += ' ORDER BY timestamp, metric_name'
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    return None
                
                # Pivot to have metrics as columns
                df['datetime'] = pd.to_datetime(df['datetime'])
                df_pivot = df.pivot_table(
                    index=['datetime', 'timestamp'],
                    columns='metric_name',
                    values='metric_value',
                    aggfunc='first'
                ).reset_index()
                
                df_pivot = df_pivot.set_index('datetime').sort_index()
                
                return df_pivot
                
        except Exception as e:
            logger.error(f"Error loading on-chain data from database: {e}")
            return None
    
    def get_combined_features(
        self,
        pair: str,
        token_symbol: str = None,
        interval: str = '1m',
        lookback: int = 100,
        network: str = 'ethereum',
        include_onchain: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get combined market data and on-chain data features.
        
        Args:
            pair: Trading pair (e.g., 'BTC/USD')
            token_symbol: Token symbol (e.g., 'BTC'). If None, extracted from pair.
            interval: Data interval
            lookback: Number of recent rows
            network: Blockchain network for on-chain data
            include_onchain: Whether to include on-chain data
        
        Returns:
            DataFrame with combined features
        """
        # Get market data
        market_features = self.get_latest_features(
            pair, interval=interval, lookback=lookback, 
            include_indicators=True, normalize=False
        )
        
        if market_features is None or market_features.empty:
            return None
        
        if not include_onchain:
            return market_features
        
        # Extract token symbol from pair if not provided
        if token_symbol is None:
            token_symbol = pair.split('/')[0]
        
        # Get on-chain data
        # For on-chain data, we typically want hourly or daily data
        onchain_interval = '1h' if interval in ['1m', '5m', '15m'] else '1d'
        
        # Get on-chain data for the same time period
        start_ts = int(market_features['timestamp'].iloc[0])
        end_ts = int(market_features['timestamp'].iloc[-1])
        
        onchain_data = self.load_onchain_data_from_db(
            token_symbol, network, 
            start_time=start_ts, end_time=end_ts
        )
        
        if onchain_data is None or onchain_data.empty:
            logger.warning(f"No on-chain data available for {token_symbol}")
            return market_features
        
        # Merge on-chain data with market data
        # Resample on-chain data to match market data frequency if needed
        if len(onchain_data) < len(market_features):
            # Forward fill on-chain data to match market data timestamps
            market_features = market_features.copy()
            for col in onchain_data.columns:
                if col not in ['timestamp']:
                    # Interpolate on-chain data to market data timestamps
                    onchain_series = onchain_data[col].reindex(
                        market_features.index, method='ffill'
                    )
                    market_features[f'onchain_{col}'] = onchain_series
        else:
            # Merge on nearest timestamp
            market_features = market_features.merge(
                onchain_data, left_on='timestamp', right_on='timestamp',
                how='left', suffixes=('', '_onchain')
            )
        
        logger.info(f"Combined {len(market_features)} market features with on-chain data")
        return market_features
    
    def start_continuous_collection(
        self, 
        pairs: List[str],
        interval: str = '1m',
        run_forever: bool = True
    ):
        """
        Start continuous data collection loop.
        
        Args:
            pairs: List of trading pairs to monitor
            interval: Data interval
            run_forever: Whether to run continuously
        """
        logger.info(f"Starting continuous data collection for {len(pairs)} pairs")
        
        while True:
            try:
                # Update market data (incremental updates)
                self.update_market_data(pairs, interval=interval, limit=1000, incremental=True)
                
                # Log status from database
                stats = self.get_database_stats()
                for pair in pairs:
                    # Get latest data
                    df = self.load_ohlcv_from_db(pair, interval, limit=1)
                    if df is not None and len(df) > 0:
                        logger.info(
                            f"{pair}: {stats.get('rows_per_pair', [])}, "
                            f"latest: {df.index[-1]}, "
                            f"price: ${df['close'].iloc[-1]:.2f}"
                        )
                    else:
                        logger.warning(f"{pair}: No data available")
                
                if not run_forever:
                    break
                
                # Wait for next update
                time.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                logger.info("Data collection stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous collection: {e}")
                time.sleep(self.update_interval)


# Example usage
if __name__ == "__main__":
    # Initialize data engine (uses config for API keys and settings)
    # API keys are loaded from environment variables or config.json
    engine = DataEngine()
    
    # Get available pairs
    pairs = engine.get_available_pairs()
    if not pairs:
        # Fallback to common pairs
        pairs = ['BTC/USD', 'ETH/USD']
        logger.info(f"Using fallback pairs: {pairs}")
    
    # Update data for selected pairs
    selected_pairs = pairs[:5]  # Use first 5 pairs for testing
    logger.info(f"Updating data for: {selected_pairs}")
    
    engine.update_market_data(selected_pairs, interval='1m', limit=1000, incremental=True)
    
    # Get database statistics
    stats = engine.get_database_stats()
    print(f"\nDatabase Statistics:")
    print(f"Total rows: {stats.get('total_rows', 0)}")
    print(f"Pairs: {stats.get('pairs', [])}")
    print(f"Intervals: {stats.get('intervals', [])}")
    
    # Get latest features with indicators
    for pair in selected_pairs:
        features = engine.get_latest_features(
            pair, 
            interval='1m',
            lookback=100, 
            include_indicators=True
        )
        if features is not None and len(features) > 0:
            print(f"\n{pair} - Latest Features:")
            print(features.tail().to_string())
            print(f"\nColumns: {list(features.columns)}")
    
    # Uncomment to start continuous collection:
    # engine.start_continuous_collection(selected_pairs, interval='1m', run_forever=True)

