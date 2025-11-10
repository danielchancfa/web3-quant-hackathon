"""
Quick tests for Horus blockchain metrics endpoints.
Verifies transaction count, mining work, chain TVL, and whale net flow endpoints.
"""

from data_pipeline.horus_api import get_transaction_count, get_mining_work, get_chain_tvl, get_whale_net_flow
import time
from datetime import datetime
from typing import Callable
import pandas as pd


def _print_basic_dataframe_info(df: pd.DataFrame, label: str) -> None:
    """Utility to print a quick summary of a DataFrame."""
    print(f"\n   {label}:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Date range: {df['datetime'].min()} -> {df['datetime'].max()}")
    print("   Sample rows:")
    print(df.head())


def _test_metric(
    fetch_func: Callable,
    metric_name: str,
    chain: str = 'bitcoin',
    interval: str = '1d'
) -> None:
    """Generic tester for Horus metric endpoints that return DataFrames."""
    print("=" * 60)
    print(f"{metric_name} — Basic Test")
    print("=" * 60)

    # 1. Basic request (full history)
    print("\n1. Fetching full history (default range)...")
    df = fetch_func(chain=chain, interval=interval, as_dataframe=True)
    if df is not None and not df.empty:
        print("   ✓ Success!")
        _print_basic_dataframe_info(df, "Full history summary")
    else:
        print("   ✗ Failed - No data returned")
        return

    # Prepare time range for historical tests
    end_time = int(time.time())
    start_time = end_time - (7 * 24 * 60 * 60)  # Last 7 days

    # 2. Historical window
    print("\n" + "-" * 60)
    print("2. Fetching historical window (last 7 days)...")
    df_hist = fetch_func(
        chain=chain,
        interval=interval,
        start=start_time,
        end=end_time,
        as_dataframe=True
    )
    if df_hist is not None and not df_hist.empty:
        print("   ✓ Success!")
        _print_basic_dataframe_info(df_hist, "Historical window summary")
    else:
        print("   ✗ Failed - Historical window returned no data")
        return

    # 3. Basic statistics
    print("\n" + "-" * 60)
    print("3. Basic statistics:")
    value_col = df_hist.columns[-1]
    print(f"   Value column: {value_col}")
    print(f"   Min value: {df_hist[value_col].min()}")
    print(f"   Max value: {df_hist[value_col].max()}")
    print(f"   Mean value: {df_hist[value_col].mean()}")


def quick_test() -> None:
    """Run quick tests for Horus endpoints."""
    print("=" * 60)
    print("Quick Tests: Horus Blockchain Metrics")
    print("=" * 60)

    _test_metric(get_transaction_count, "Transaction Count (ZH)")
    _test_metric(get_mining_work, "Mining Work (ZH)")

    chains = ['ethereum', 'solana', 'tron', 'bitcoin', 'bsc']
    for ch in chains:
        _test_metric(get_chain_tvl, f"Chain TVL (USD) — {ch}", chain=ch)

    # Whale net flow focuses primarily on Bitcoin
    print("\n" + "=" * 60)
    print("Whale Net Flow — Bitcoin")
    print("=" * 60)
    _test_metric(get_whale_net_flow, "Whale Net Flow (Native Token)", chain='bitcoin')

    # Also demonstrate fetching all chains at once
    print("\n" + "=" * 60)
    print("Chain TVL — All Chains (Combined DataFrame)")
    print("=" * 60)
    df_all = get_chain_tvl(chains, interval='1d', as_dataframe=True)
    if df_all is not None and not df_all.empty:
        print(f"Combined shape: {df_all.shape}")
        print(f"Chains: {df_all['chain'].unique()}")
        print(df_all.head())
    else:
        print("Failed to fetch combined chain TVL data")

    print("\n" + "=" * 60)
    print("All tests completed. Review outputs above.")
    print("=" * 60)


if __name__ == "__main__":
    quick_test()
