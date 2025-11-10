"""
Quick test for CoinMarketCap Fear & Greed API
--------------------------------------------
"""

from data_pipeline.coinmarketcap_api import get_fear_and_greed_historical


def quick_test():
    print("=" * 60)
    print("CoinMarketCap Fear & Greed Quick Test")
    print("=" * 60)

    # Basic fetch
    print("\nFetching available history...")
    df = get_fear_and_greed_historical(as_dataframe=True)
    if df is not None and not df.empty:
        print("   ✓ Success!")
        print(f"   Shape: {df.shape}")
        print(f"   Date range: {df['datetime'].min()} -> {df['datetime'].max()}")
        print(df.head())
    else:
        print("   ✗ Failed to fetch data")


if __name__ == "__main__":
    quick_test()
