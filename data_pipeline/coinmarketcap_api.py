"""
CoinMarketCap API Client
------------------------
Provides helpers to fetch data from the CoinMarketCap APIs.

Docs: https://pro-api.coinmarketcap.com/
"""

import requests
import logging
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
import pandas as pd

from config import get_coinmarketcap_api_key

logger = logging.getLogger(__name__)

CMC_BASE_URL = "https://pro-api.coinmarketcap.com"


def _get_cmc_headers() -> Dict[str, str]:
    """Return headers for CoinMarketCap API requests."""
    api_key = get_coinmarketcap_api_key()
    headers = {
        'Accepts': 'application/json'
    }
    if api_key:
        headers['X-CMC_PRO_API_KEY'] = api_key
    else:
        logger.warning("CoinMarketCap API key not set. Set COINMARKETCAP_API_KEY.")
    return headers


def _format_cmc_time(value: Union[str, int, float, datetime]) -> str:
    """Format various time inputs for CoinMarketCap (ISO 8601 string)."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        return value
    raise TypeError(f"Unsupported time format for CoinMarketCap request: {type(value)}")


def get_ohlcv_historical(
    symbol: str,
    interval: str = 'daily',
    convert: str = 'USD',
    start: Optional[Union[str, int, float, datetime]] = None,
    end: Optional[Union[str, int, float, datetime]] = None,
    count: Optional[int] = None,
    as_dataframe: bool = False
) -> Optional[Union[Dict[str, Any], pd.DataFrame]]:
    """Fetch historical OHLCV data for a symbol.

    Endpoint: https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical

    Args:
        symbol: Asset ticker (e.g., 'BTC')
        interval: 'hourly', 'daily'
        convert: Quote currency, default USD
        start: Optional start time (ISO string, datetime, or timestamp seconds)
        end: Optional end time (ISO string, datetime, or timestamp seconds)
        count: Optional max number of data points to return (<= 5000 per CMC docs)
        as_dataframe: Return pandas DataFrame if True.

    Returns:
        Raw JSON or normalized DataFrame. None on failure.
    """
    url = f"{CMC_BASE_URL}/v1/cryptocurrency/ohlcv/historical"
    headers = _get_cmc_headers()
    time_period = 'daily'
    if isinstance(interval, str) and interval.endswith('h'):
        time_period = 'hourly'
    elif interval in {'daily', 'weekly', 'monthly'}:
        time_period = interval

    params: Dict[str, Any] = {
        'symbol': symbol.upper(),
        'interval': interval,
        'time_period': time_period,
        'convert': convert.upper(),
    }
    if start:
        params['time_start'] = _format_cmc_time(start)
    if end:
        params['time_end'] = _format_cmc_time(end)
    if count:
        params['count'] = int(count)

    try:
        res = requests.get(url, headers=headers, params=params, timeout=60)
        res.raise_for_status()
        data = res.json()

        if as_dataframe:
            df = _normalize_ohlcv_data(data, symbol=symbol.upper(), interval=interval, convert=convert.upper())
            return df
        return data

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching OHLCV for {symbol}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching OHLCV for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching OHLCV for {symbol}: {e}")
        return None


def get_top_market_cap_symbols(
    limit: int = 20,
    convert: str = 'USD'
) -> Optional[List[str]]:
    """Fetch top market-cap crypto symbols from CoinMarketCap.

    Endpoint: https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest

    Args:
        limit: Maximum number of symbols to return.
        convert: Quote currency for market cap ranking (default USD).

    Returns:
        List of uppercase symbols ordered by market cap, or None on failure.
    """
    url = f"{CMC_BASE_URL}/v1/cryptocurrency/listings/latest"
    headers = _get_cmc_headers()
    params = {
        'start': 1,
        'limit': limit,
        'convert': convert.upper(),
        'sort': 'market_cap',
        'sort_dir': 'desc'
    }

    try:
        res = requests.get(url, headers=headers, params=params, timeout=30)
        res.raise_for_status()
        payload = res.json()
        data = payload.get('data')
        if not isinstance(data, list):
            logger.warning("Unexpected response format for CoinMarketCap listings.")
            return None
        symbols = [entry.get('symbol', '').upper() for entry in data if entry.get('symbol')]
        return symbols[:limit]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching top market cap symbols from CoinMarketCap: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching top market cap symbols: {e}")
        return None


def get_fear_and_greed_historical(
    as_dataframe: bool = False
) -> Optional[Union[Dict[str, Any], pd.DataFrame]]:
    """Fetch historical Fear & Greed index values.

    Endpoint: https://pro-api.coinmarketcap.com/v3/fear-and-greed/historical

    Args:
        as_dataframe: If True, return a pandas DataFrame

    Returns:
        Raw JSON (dict) or pandas DataFrame if as_dataframe=True. None on failure.
    """
    url = f"{CMC_BASE_URL}/v3/fear-and-greed/historical"
    headers = _get_cmc_headers()

    try:
        res = requests.get(url, headers=headers, params={}, timeout=30)
        res.raise_for_status()
        data = res.json()

        if as_dataframe:
            df = _normalize_fear_greed_data(data)
            return df
        return data

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching fear & greed historical data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching fear & greed historical data: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching fear & greed historical data: {e}")
        return None


def _normalize_fear_greed_data(
    data: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> pd.DataFrame:
    """Normalize fear & greed API response into a DataFrame."""
    if data is None:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []

    # Expected structures: {'data': {...}} or list of points
    if isinstance(data, dict):
        payload = data.get('data', data)
        if isinstance(payload, dict):
            if 'points' in payload and isinstance(payload['points'], list):
                records = payload['points']
            elif 'values' in payload and isinstance(payload['values'], list):
                records = payload['values']
        elif isinstance(payload, list):
            records = payload
    elif isinstance(data, list):
        records = data

    if not records:
        logger.warning("Fear & Greed data had unexpected format or was empty.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Handle timestamp column (may be 'timestamp' or 'time' etc.)
    timestamp_col = None
    for candidate in ['timestamp', 'time', 'date']:
        if candidate in df.columns:
            timestamp_col = candidate
            break

    if timestamp_col is None:
        logger.warning("Fear & Greed data missing timestamp field.")
        return df

    df['timestamp'] = pd.to_numeric(df[timestamp_col], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = df['timestamp'].astype(int)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

    # Determine value field (score/value)
    value_col = None
    for candidate in ['value', 'score', 'fear_greed', 'index']:
        if candidate in df.columns:
            value_col = candidate
            break

    if value_col is None:
        # Sometimes value may be under 'data' key inside each record
        if 'data' in df.columns:
            df['value'] = pd.to_numeric(df['data'], errors='coerce')
            value_col = 'value'
        else:
            logger.warning("Fear & Greed data missing value field.")
            df['value'] = pd.NA
            value_col = 'value'

    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.rename(columns={value_col: 'fear_greed_value'})

    # Optional classification field
    classification_col = None
    for candidate in ['score_classification', 'value_classification', 'classification']:
        if candidate in df.columns:
            classification_col = candidate
            break

    if classification_col:
        df = df.rename(columns={classification_col: 'fear_greed_class'})
    else:
        df['fear_greed_class'] = pd.NA

    df['interval'] = pd.NA
    df = df.sort_values('datetime').reset_index(drop=True)

    column_order = ['datetime', 'timestamp', 'interval', 'fear_greed_value', 'fear_greed_class']
    other_cols = [col for col in df.columns if col not in column_order]
    df = df[column_order + other_cols]

    return df


def _normalize_ohlcv_data(
    data: Dict[str, Any],
    symbol: str,
    interval: str,
    convert: str
) -> pd.DataFrame:
    """Normalize CoinMarketCap OHLCV response into a DataFrame."""
    if not data or 'data' not in data:
        logger.warning(f"OHLCV data for {symbol} returned empty payload.")
        return pd.DataFrame()

    payload = data['data']
    quotes = payload.get('quotes', [])
    if not isinstance(quotes, list) or not quotes:
        logger.warning(f"OHLCV data for {symbol} missing quotes array.")
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for entry in quotes:
        quote = entry.get('quote', {}).get(convert)
        if not quote:
            continue
        time_open = entry.get('time_open')
        if not time_open:
            continue
        dt = pd.to_datetime(time_open, utc=True)
        rows.append({
            'symbol': symbol,
            'interval': interval,
            'datetime': dt,
            'timestamp': int(dt.timestamp()),
            'open': quote.get('open'),
            'high': quote.get('high'),
            'low': quote.get('low'),
            'close': quote.get('close'),
            'volume': quote.get('volume') or quote.get('volume_traded'),
            'market_cap': quote.get('market_cap')
        })

    if not rows:
        logger.warning(f"OHLCV data for {symbol} produced no usable rows.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    if df.empty:
        return df

    for col in ['open', 'high', 'low', 'close', 'volume', 'market_cap']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert(None)
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Quick demo
    df = get_fear_and_greed_historical(as_dataframe=True)
    if df is not None and not df.empty:
        print(df.head())
    else:
        print("Failed to fetch fear & greed data.")
