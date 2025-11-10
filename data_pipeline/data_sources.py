"""
Data Sources - Alternative Market Data Providers
-------------------------------------------------
Provides fallback data sources when primary API (Roostoo) is unavailable or rate-limited.
Supports multiple cryptocurrency data providers including Binance, CoinGecko, etc.
"""

import requests
import time
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BinanceDataSource:
    """
    Binance Public API data source.
    No API key required for public market data endpoints.
    """
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    # Mapping from common trading pairs to Binance symbols
    # Roostoo format: "BTC/USD" -> Binance format: "BTCUSDT"
    PAIR_MAPPING = {
        'BTC/USD': 'BTCUSDT',
        'ETH/USD': 'ETHUSDT',
        'BNB/USD': 'BNBUSDT',
        'SOL/USD': 'SOLUSDT',
        'ADA/USD': 'ADAUSDT',
        'XRP/USD': 'XRPUSDT',
        'DOT/USD': 'DOTUSDT',
        'DOGE/USD': 'DOGEUSDT',
        'MATIC/USD': 'MATICUSDT',
        'AVAX/USD': 'AVAXUSDT',
        'LINK/USD': 'LINKUSDT',
        'UNI/USD': 'UNIUSDT',
        'LTC/USD': 'LTCUSDT',
        'ATOM/USD': 'ATOMUSDT',
        'ETC/USD': 'ETCUSDT',
        'ALGO/USD': 'ALGOUSDT',
        'XLM/USD': 'XLMUSDT',
        'VET/USD': 'VETUSDT',
        'FIL/USD': 'FILUSDT',
        'TRX/USD': 'TRXUSDT',
    }
    
    @staticmethod
    def convert_pair_to_binance_symbol(pair: str) -> Optional[str]:
        """
        Convert Roostoo pair format to Binance symbol format.
        
        Args:
            pair: Trading pair in format "BTC/USD" or "BTC/USDT"
        
        Returns:
            Binance symbol like "BTCUSDT" or None if not mappable
        """
        # Normalize pair format
        pair = pair.strip().upper()
        
        # Check direct mapping first
        if pair in BinanceDataSource.PAIR_MAPPING:
            return BinanceDataSource.PAIR_MAPPING[pair]
        
        # Try to convert format: "BTC/USD" -> "BTCUSDT"
        if '/' in pair:
            parts = pair.split('/')
            if len(parts) == 2:
                base, quote = parts
                # Convert USD to USDT for Binance
                if quote == 'USD':
                    quote = 'USDT'
                symbol = f"{base}{quote}"
                return symbol
        
        # If already in Binance format (no slash), try to fix USD -> USDT
        if 'USD' in pair and '/' not in pair:
            # Handle cases like "BTCUSD" -> "BTCUSDT"
            if pair.endswith('USD') and not pair.endswith('USDT'):
                return pair.replace('USD', 'USDT')
            return pair
        
        return None
    
    @staticmethod
    def is_symbol_available_on_binance(symbol: str) -> bool:
        """
        Check if a symbol is available on Binance.
        
        Args:
            symbol: Binance symbol (e.g., 'BTCUSDT')
        
        Returns:
            True if symbol exists on Binance
        """
        try:
            info = BinanceDataSource.get_exchange_info()
            if info and 'symbols' in info:
                symbols = [s['symbol'] for s in info['symbols']]
                return symbol in symbols
        except Exception as e:
            logger.warning(f"Error checking Binance symbol availability: {e}")
        return False
    
    @staticmethod
    def get_klines(
        symbol: str,
        interval: str = '1m',
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Optional[List]:
        """
        Get kline/candlestick data from Binance.
        
        Args:
            symbol: Binance symbol (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of klines to return (default 500, max 1000)
            start_time: Start time in milliseconds (optional)
            end_time: End time in milliseconds (optional)
        
        Returns:
            List of klines: [[open_time, open, high, low, close, volume, close_time, ...], ...]
        """
        url = f"{BinanceDataSource.BASE_URL}/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1000)  # Binance max is 1000
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        try:
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Binance API error for {symbol}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.warning(f"Response: {e.response.text}")
            return None
    
    @staticmethod
    def get_24h_ticker(symbol: str) -> Optional[Dict]:
        """
        Get 24hr ticker price statistics from Binance.
        
        Args:
            symbol: Binance symbol (e.g., 'BTCUSDT')
        
        Returns:
            Dict with ticker data
        """
        url = f"{BinanceDataSource.BASE_URL}/ticker/24hr"
        params = {'symbol': symbol}
        
        try:
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Binance ticker error for {symbol}: {e}")
            return None
    
    @staticmethod
    def get_exchange_info() -> Optional[Dict]:
        """
        Get Binance exchange information including all trading pairs.
        
        Returns:
            Dict with exchange info
        """
        url = f"{BinanceDataSource.BASE_URL}/exchangeInfo"
        
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Binance exchange info error: {e}")
            return None
    
    @staticmethod
    def fetch_klines_for_pair(
        pair: str,
        interval: str = '1m',
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch klines for a Roostoo-format pair using Binance.
        
        Args:
            pair: Trading pair in Roostoo format (e.g., 'BTC/USD')
            interval: Kline interval
            limit: Number of klines
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
        
        Returns:
            DataFrame with OHLCV data, or None if failed
        """
        # Convert pair to Binance symbol
        symbol = BinanceDataSource.convert_pair_to_binance_symbol(pair)
        if not symbol:
            logger.warning(f"Cannot convert pair {pair} to Binance symbol")
            return None
        
        # Fetch klines
        klines = BinanceDataSource.get_klines(
            symbol, interval, limit, start_time, end_time
        )
        
        if not klines:
            return None
        
        # Convert to DataFrame
        # Binance format: [open_time, open, high, low, close, volume, close_time, ...]
        df_data = []
        for kline in klines:
            df_data.append({
                'timestamp': int(kline[0]),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5])
            })
        
        if not df_data:
            return None
        
        df = pd.DataFrame(df_data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime').sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        logger.info(f"Fetched {len(df)} klines from Binance for {pair} ({symbol})")
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


class MultiSourceDataFetcher:
    """
    Multi-source data fetcher with fallback support.
    Tries Roostoo first, falls back to alternative sources if needed.
    """
    
    def __init__(self, primary_source='roostoo', fallback_sources=None):
        """
        Initialize multi-source fetcher.
        
        Args:
            primary_source: Primary data source ('roostoo')
            fallback_sources: List of fallback sources (default: ['binance'])
        """
        self.primary_source = primary_source
        self.fallback_sources = fallback_sources or ['binance']
        
        logger.info(f"MultiSourceDataFetcher initialized: primary={primary_source}, fallbacks={fallback_sources}")
    
    def fetch_klines(
        self,
        pair: str,
        interval: str = '1m',
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        use_roostoo: bool = True,
        use_fallback: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch klines from primary or fallback sources.
        
        Args:
            pair: Trading pair (e.g., 'BTC/USD')
            interval: Kline interval
            limit: Number of klines
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            use_roostoo: Whether to try Roostoo first
            use_fallback: Whether to use fallback if primary fails
        
        Returns:
            DataFrame with OHLCV data, or None if all sources fail
        """
        # Try Roostoo first
        if use_roostoo:
            try:
                from .roostoo_api import get_klines
                response = get_klines(pair, interval=interval, limit=limit, 
                                     start_time=start_time, end_time=end_time)
                
                # Check for 429 rate limit error
                if response is None:
                    # Check if it's a 429 error by trying to get error details
                    logger.warning(f"Roostoo returned None for {pair}, may be rate limited")
                else:
                    # Parse Roostoo response (same logic as in data_engine)
                    df = self._parse_roostoo_response(response, pair)
                    if df is not None and not df.empty:
                        logger.info(f"Successfully fetched {len(df)} klines from Roostoo for {pair}")
                        return df
                    else:
                        logger.warning(f"Roostoo response for {pair} was empty or invalid")
            except requests.exceptions.HTTPError as e:
                if hasattr(e, 'response') and e.response is not None:
                    if e.response.status_code == 429:
                        logger.warning(f"Roostoo rate limited (429) for {pair}, using fallback")
                    else:
                        logger.warning(f"Roostoo HTTP error {e.response.status_code} for {pair}: {e}")
                else:
                    logger.warning(f"Roostoo fetch failed for {pair}: {e}")
            except Exception as e:
                logger.warning(f"Roostoo fetch failed for {pair}: {e}")
        
        # Try fallback sources
        if use_fallback:
            for source in self.fallback_sources:
                try:
                    if source == 'binance':
                        # Check if pair can be converted to Binance symbol
                        symbol = BinanceDataSource.convert_pair_to_binance_symbol(pair)
                        if symbol is None:
                            logger.warning(f"Cannot convert {pair} to Binance symbol, skipping Binance fallback")
                            continue
                        
                        logger.info(f"Trying Binance fallback for {pair} (symbol: {symbol})")
                        df = BinanceDataSource.fetch_klines_for_pair(
                            pair, interval, limit, start_time, end_time
                        )
                        if df is not None and not df.empty:
                            logger.info(f"Successfully fetched {len(df)} klines from Binance for {pair}")
                            return df
                        else:
                            logger.warning(f"Binance returned empty data for {pair}")
                except Exception as e:
                    logger.warning(f"Fallback source {source} failed for {pair}: {e}")
                    continue
        
        logger.error(f"All data sources failed for {pair}")
        return None
    
    def _parse_roostoo_response(self, response, pair: str) -> Optional[pd.DataFrame]:
        """
        Parse Roostoo API response into DataFrame.
        
        Args:
            response: API response
            pair: Trading pair (for logging)
        
        Returns:
            DataFrame or None
        """
        try:
            # Parse response - format may vary
            if isinstance(response, dict):
                if 'Data' in response:
                    data = response['Data']
                elif 'data' in response:
                    data = response['data']
                elif 'klines' in response:
                    data = response['klines']
                else:
                    data = list(response.values())[0] if response else []
            elif isinstance(response, list):
                data = response
            else:
                return None
            
            if not data or len(data) == 0:
                return None
            
            # Convert to DataFrame
            df_data = []
            for kline in data:
                if isinstance(kline, list) and len(kline) >= 6:
                    df_data.append({
                        'timestamp': int(kline[0]),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    })
                elif isinstance(kline, dict):
                    df_data.append({
                        'timestamp': int(kline.get('openTime', kline.get('timestamp', 0))),
                        'open': float(kline.get('open', 0)),
                        'high': float(kline.get('high', 0)),
                        'low': float(kline.get('low', 0)),
                        'close': float(kline.get('close', 0)),
                        'volume': float(kline.get('volume', 0))
                    })
            
            if not df_data:
                return None
            
            df = pd.DataFrame(df_data)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('datetime').sort_index()
            df = df[~df.index.duplicated(keep='last')]
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Error parsing Roostoo response: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Test Binance data source
    print("Testing Binance Data Source...")
    
    # Test pair conversion
    test_pairs = ['BTC/USD', 'ETH/USD', 'SOL/USD']
    for pair in test_pairs:
        symbol = BinanceDataSource.convert_pair_to_binance_symbol(pair)
        print(f"{pair} -> {symbol}")
    
    # Test fetching data
    print("\nFetching test data from Binance...")
    df = BinanceDataSource.fetch_klines_for_pair('BTC/USD', interval='1m', limit=10)
    if df is not None:
        print(f"Successfully fetched {len(df)} rows")
        print(df.head())
    else:
        print("Failed to fetch data")
    
    # Test multi-source fetcher
    print("\nTesting MultiSourceDataFetcher...")
    fetcher = MultiSourceDataFetcher()
    df = fetcher.fetch_klines('BTC/USD', interval='1m', limit=10, use_roostoo=False)
    if df is not None:
        print(f"Successfully fetched {len(df)} rows from fallback source")
        print(df.head())

