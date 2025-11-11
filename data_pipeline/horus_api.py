"""
Horus API Client - On-Chain Data Provider
------------------------------------------
Integrates with Horus API to fetch on-chain cryptocurrency data.

API Documentation: https://api-horus.com/docs
Base URL: https://api-horus.com
"""

import requests
from typing import Optional, Dict, Union, List, Any
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Horus API Configuration
HORUS_BASE_URL = "https://api-horus.com"

# Import configuration
from config import get_config


def set_horus_api_key(api_key: str):
    """Set Horus API key (runtime override)."""
    get_config().set('horus_api_key', api_key)
    logger.info("Horus API key set (runtime override)")


def get_horus_api_key() -> Optional[str]:
    """Get Horus API key from config (environment variables or config file)."""
    return get_config().horus_api_key


def _get_headers() -> Dict[str, str]:
    """Get headers for Horus API requests."""
    api_key = get_horus_api_key()
    headers = {

    }
    if api_key:
        headers['X-API-Key'] = f'{api_key}'
    else:
        logger.warning("Horus API key not set. Some endpoints may require authentication.")
    return headers


def get_transaction_count(
    chain: str,
    interval: str = '1d',
    start: Optional[int] = None,
    end: Optional[int] = None,
    as_dataframe: bool = False
) -> Optional[Union[Dict[str, Any], pd.DataFrame]]:
    """
    Get transaction count for a blockchain network.
    
    Endpoint: https://api-horus.com/blockchain/transaction_count
    
    Args:
        chain: Blockchain chain name (e.g., 'bitcoin', 'ethereum')
        interval: Time interval for aggregation (default: '1d')
        start: Start timestamp in seconds (inclusive), optional
        end: End timestamp in seconds (exclusive), optional
        as_dataframe: If True, return pandas DataFrame with normalized data
    
    Returns:
        - If as_dataframe=True: pandas DataFrame with transaction counts
        - Otherwise: Raw JSON response as a Python dict
        - None if request fails
    """
    url = f"{HORUS_BASE_URL}/blockchain/transaction_count"
    
    headers = _get_headers()
    params: Dict[str, Any] = {}

    # Include API key as query parameter if required
    if 'X-API-Key' in headers:
        params['X-API-Key'] = headers['X-API-Key']

    params['chain'] = chain
    params['interval'] = interval
    
    # Add optional parameters
    if start is not None:
        params['start'] = start
    if end is not None:
        params['end'] = end
    
    try:
        res = requests.get(url, headers=headers, params=params, timeout=30)
        res.raise_for_status()
        data = res.json()

        if as_dataframe:
            df = _normalize_timeseries_data(
                data=data,
                chain=chain,
                interval=interval,
                value_field='transaction_count',
                value_column_name='transaction_count'
            )
            return df

        return data
            
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching transaction count for {chain}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching transaction count for {chain}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching transaction count for {chain}: {e}")
        return None


def get_mining_work(
    chain: str,
    interval: str = '1d',
    start: Optional[int] = None,
    end: Optional[int] = None,
    as_dataframe: bool = False
) -> Optional[Union[Dict[str, Any], pd.DataFrame]]:
    """
    Get average mining work required to mine a block (in zettahashes).

    Endpoint: https://api-horus.com/blockchain/mining_work

    Args:
        chain: Blockchain chain name (e.g., 'bitcoin')
        interval: Time interval for aggregation (default: '1d')
        start: Start timestamp in seconds (inclusive), optional
        end: End timestamp in seconds (exclusive), optional
        as_dataframe: If True, return pandas DataFrame

    Returns:
        - If as_dataframe=True: pandas DataFrame with mining work (ZH)
        - Otherwise: Raw JSON response as a Python dict
        - None if request fails
    """
    url = f"{HORUS_BASE_URL}/blockchain/mining_work"

    headers = _get_headers()
    params: Dict[str, Any] = {}

    if 'X-API-Key' in headers:
        params['X-API-Key'] = headers['X-API-Key']

    params['chain'] = chain
    params['interval'] = interval
    if start is not None:
        params['start'] = start
    if end is not None:
        params['end'] = end

    try:
        res = requests.get(url, headers=headers, params=params, timeout=30)
        res.raise_for_status()

        data = res.json()

        if as_dataframe:
            df = _normalize_timeseries_data(
                data=data,
                chain=chain,
                interval=interval,
                value_field='work_zh',
                value_column_name='mining_work'
            )
            return df

        return data

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching mining work for {chain}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching mining work for {chain}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching mining work for {chain}: {e}")
        return None


def get_chain_tvl(
    chain: Union[str, List[str]],
    interval: str = '1d',
    start: Optional[int] = None,
    end: Optional[int] = None,
    as_dataframe: bool = False
) -> Optional[Union[Dict[str, Any], pd.DataFrame]]:
    """
    Get chain TVL (Total Value Locked) in USD.

    Endpoint: https://api-horus.com/blockchain/chain_tvl

    Args:
        chain: Chain name or list of chain names ('ethereum', 'solana', 'tron', 'bitcoin', 'bsc', 'base')
        interval: Time interval for aggregation (default: '1d')
        start: Start timestamp in seconds (inclusive), optional
        end: End timestamp in seconds (exclusive), optional
        as_dataframe: If True, return pandas DataFrame

    Returns:
        - If as_dataframe=True: pandas DataFrame with TVL data
        - Otherwise: Raw JSON response (dict for single chain, dict of dicts for multiple chains)
        - None if request fails
    """
    url = f"{HORUS_BASE_URL}/defi/chain_tvl"
    
    headers = _get_headers()

    if isinstance(chain, str):
        chain_list = [chain]
        single_chain = True
    else:
        chain_list = list(chain)
        single_chain = False

    data_frames: List[pd.DataFrame] = []
    raw_results: Dict[str, Any] = {}

    for ch in chain_list:
        params: Dict[str, Any] = {}
        if 'X-API-Key' in headers:
            params['X-API-Key'] = headers['X-API-Key']

        params['chain'] = ch
        params['interval'] = interval

        if start is not None:
            params['start'] = start
        if end is not None:
            params['end'] = end

        try:
            res = requests.get(url, headers=headers, params=params, timeout=30)
            res.raise_for_status()
            data = res.json()

            if as_dataframe:
                df = _normalize_timeseries_data(
                    data=data,
                    chain=ch,
                    interval=interval,
                    value_field='tvl',
                    value_column_name='tvl'
                )
                if df is not None and not df.empty:
                    data_frames.append(df)
            else:
                raw_results[ch] = data

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching chain TVL for {ch}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching chain TVL for {ch}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching chain TVL for {ch}: {e}")
            return None

    if as_dataframe:
        if not data_frames:
            return pd.DataFrame()
        combined_df = pd.concat(data_frames, ignore_index=True)
        combined_df = combined_df.sort_values(['chain', 'datetime']).reset_index(drop=True)
        return combined_df
    else:
        return raw_results[chain_list[0]] if single_chain else raw_results


def get_whale_net_flow(
    chain: Union[str, List[str]],
    interval: str = '1d',
    start: Optional[int] = None,
    end: Optional[int] = None,
    as_dataframe: bool = False
) -> Optional[Union[Dict[str, Any], pd.DataFrame]]:
    """
    Get whale net flow (net change in whale balances) in native tokens.

    Endpoint: https://api-horus.com/addresses/whale_net_flow

    Args:
        chain: Chain name or list of chain names (e.g., 'bitcoin')
        interval: Time interval for aggregation (default: '1d')
        start: Start timestamp in seconds (inclusive), optional
        end: End timestamp in seconds (exclusive), optional
        as_dataframe: If True, return pandas DataFrame

    Returns:
        - If as_dataframe=True: pandas DataFrame with whale net flow data
        - Otherwise: Raw JSON response (dict or dict of dicts for multiple chains)
        - None if request fails
    """
    url = f"{HORUS_BASE_URL}/addresses/whale_net_flow"
    headers = _get_headers()

    if isinstance(chain, str):
        chain_list = [chain]
        single_chain = True
    else:
        chain_list = list(chain)
        single_chain = False

    data_frames: List[pd.DataFrame] = []
    raw_results: Dict[str, Any] = {}

    for ch in chain_list:
        params: Dict[str, Any] = {}
        if 'X-API-Key' in headers:
            params['X-API-Key'] = headers['X-API-Key']

        params['chain'] = ch
        params['interval'] = interval

        if start is not None:
            params['start'] = start
        if end is not None:
            params['end'] = end

        try:
            res = requests.get(url, headers=headers, params=params, timeout=30)
            res.raise_for_status()
            data = res.json()

            if as_dataframe:
                df = _normalize_timeseries_data(
                    data=data,
                    chain=ch,
                    interval=interval,
                    value_field='whale_net_flow',
                    value_column_name='whale_net_flow'
                )
                if df is not None and not df.empty:
                    data_frames.append(df)
            else:
                raw_results[ch] = data

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching whale net flow for {ch}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching whale net flow for {ch}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching whale net flow for {ch}: {e}")
            return None

    if as_dataframe:
        if not data_frames:
            return pd.DataFrame()
        combined_df = pd.concat(data_frames, ignore_index=True)
        combined_df = combined_df.sort_values(['chain', 'datetime']).reset_index(drop=True)
        return combined_df
    else:
        return raw_results[chain_list[0]] if single_chain else raw_results


def get_market_price(
    asset: str,
    interval: str = "1d",
    start: Optional[int] = None,
    end: Optional[int] = None,
    as_dataframe: bool = False,
) -> Optional[Union[List[Dict[str, Any]], pd.DataFrame]]:
    """
    Fetch historical market price for an asset.

    Endpoint: https://api-horus.com/market/price
    Supported intervals: '1d', '1h', '15m'
    """
    url = f"{HORUS_BASE_URL}/market/price"
    headers = _get_headers()
    params: Dict[str, Any] = {
        "asset": asset.upper(),
        "interval": interval,
    }
    if 'X-API-Key' in headers:
        params['X-API-Key'] = headers['X-API-Key']
    if start is not None:
        params["start"] = int(start)
    if end is not None:
        params["end"] = int(end)

    try:
        res = requests.get(url, headers=headers, params=params, timeout=30)
        res.raise_for_status()
        data = res.json()

        if as_dataframe:
            df = pd.DataFrame(data)
            if df.empty:
                return pd.DataFrame()
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df.dropna(subset=['timestamp', 'price'])
            if df.empty:
                return pd.DataFrame()
            df['timestamp'] = df['timestamp'].astype(int)
            df['price'] = df['price'].astype(float)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            df['asset'] = asset.upper()
            df['interval'] = interval
            df = df.sort_values('datetime').reset_index(drop=True)
            return df[['asset', 'interval', 'timestamp', 'datetime', 'price']]

        return data

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching market price for {asset}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching market price for {asset}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching market price for {asset}: {e}")
        return None


def _normalize_timeseries_data(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    chain: str,
    interval: str,
    value_field: str,
    value_column_name: str
) -> pd.DataFrame:
    """
    Normalize Horus time-series data into a pandas DataFrame.

    Args:
        data: Raw data returned from Horus API
        chain: Blockchain chain name
        interval: Time interval used for the query
        value_field: Field name in API response that contains the metric value
        value_column_name: Column name to use in the DataFrame

    Returns:
        pandas DataFrame standardized with columns:
            - datetime (datetime64[ns, UTC])
            - timestamp (int, seconds)
            - chain (str)
            - interval (str)
            - <value_column_name> (numeric)
        Additional fields from the API response are preserved.
    """
    if data is None:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []

    # Case 1: Data is a dict with 'data' key
    if isinstance(data, dict):
        if 'data' in data and isinstance(data['data'], list):
            if all(isinstance(item, dict) for item in data['data']):
                records = data['data']
        # Case 2: Dict mapping timestamps to counts
        elif all(
            isinstance(key, (str, int)) and isinstance(val, (int, float))
            for key, val in data.items()
        ):
            for key, value in data.items():
                try:
                    ts = int(key)
                except ValueError:
                    # Attempt to parse ISO timestamp
                    try:
                        ts = int(pd.to_datetime(key, utc=True).timestamp())
                    except Exception:
                        continue
                records.append({
                    'timestamp': ts,
                    'transaction_count': value
                })
        # Case 3: Already a record dict (single entry)
        elif 'timestamp' in data and value_field in data:
            records = [data]
        else:
            logger.warning("Unexpected time-series data structure (dict).")
    # Case 4: Data is a list of records
    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            records = data
        else:
            logger.warning("Unexpected time-series data structure (list).")
    else:
        logger.warning(f"Unsupported data type for time-series: {type(data)}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if 'timestamp' not in df.columns:
        logger.warning("Time-series data missing 'timestamp' field.")
        return df

    # Clean timestamp
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = df['timestamp'].astype(int)

    # Convert to datetime (UTC)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    
    # Ensure value field is numeric
    if value_field in df.columns:
        df[value_field] = pd.to_numeric(df[value_field], errors='coerce')
    else:
        logger.warning(f"Time-series data missing '{value_field}' field.")

    # Rename value field to standard column name
    if value_field in df.columns:
        df = df.rename(columns={value_field: value_column_name})

    # Add metadata
    df['chain'] = chain
    df['interval'] = interval

    # Sort chronologically
    df = df.sort_values('datetime').reset_index(drop=True)

    # Reorder columns for clarity
    column_order = [
        'datetime', 'timestamp', 'chain', 'interval', value_column_name
    ] + [col for col in df.columns if col not in [
        'datetime', 'timestamp', 'chain', 'interval', value_column_name
    ]]
    df = df[column_order]

    return df


# Example usage
if __name__ == "__main__":
    import os
    import time

    # Set API key if available
    api_key = os.getenv('HORUS_API_KEY')
    if api_key:
        set_horus_api_key(api_key)

    print("=== Horus API Quick Demo ===")

    # Transaction count demo
    print("\nTransaction Count (JSON -> DataFrame):")
    tx_df = get_transaction_count(chain='bitcoin', interval='1d', as_dataframe=True)
    if tx_df is not None and not tx_df.empty:
        print(tx_df.head())
    else:
        print("Failed to fetch transaction count data.")

    # Mining work demo
    print("\nMining Work (JSON -> DataFrame):")
    mining_df = get_mining_work(chain='bitcoin', interval='1d', as_dataframe=True)
    if mining_df is not None and not mining_df.empty:
        print(mining_df.head())
    else:
        print("Failed to fetch mining work data.")

    # Historical window example
    print("\nHistorical window (last 7 days):")
    end_time = int(time.time())
    start_time = end_time - (7 * 24 * 60 * 60)
    tx_hist = get_transaction_count(
        chain='bitcoin',
        interval='1d',
        start=start_time,
        end=end_time,
        as_dataframe=True
    )
    if tx_hist is not None and not tx_hist.empty:
        print(tx_hist.tail())