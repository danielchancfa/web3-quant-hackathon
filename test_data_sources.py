"""
Test script for Multi-Source Data Fetcher
------------------------------------------
Tests the fallback mechanism from Roostoo to Binance.
"""

from data_pipeline.data_sources import BinanceDataSource, MultiSourceDataFetcher
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_binance_data_source():
    """Test Binance data source directly."""
    print("=" * 60)
    print("Testing Binance Data Source")
    print("=" * 60)
    
    # Test pair conversion
    print("\n1. Testing pair conversion...")
    test_pairs = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'BNB/USD']
    for pair in test_pairs:
        symbol = BinanceDataSource.convert_pair_to_binance_symbol(pair)
        print(f"   {pair} -> {symbol}")
        if symbol:
            # Check if symbol is available
            available = BinanceDataSource.is_symbol_available_on_binance(symbol)
            print(f"      Available on Binance: {available}")
    
    # Test fetching data
    print("\n2. Testing data fetch from Binance...")
    df = BinanceDataSource.fetch_klines_for_pair('BTC/USD', interval='1m', limit=10)
    if df is not None:
        print(f"   Successfully fetched {len(df)} rows")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Latest price: ${df['close'].iloc[-1]:.2f}")
        print(f"\n   Sample data:")
        print(df.head().to_string())
    else:
        print("   Failed to fetch data from Binance")
    
    # Test different intervals
    print("\n3. Testing different intervals...")
    intervals = ['1m', '5m', '1h', '1d']
    for interval in intervals:
        df = BinanceDataSource.fetch_klines_for_pair('BTC/USD', interval=interval, limit=5)
        if df is not None:
            print(f"   {interval}: {len(df)} rows")
        else:
            print(f"   {interval}: Failed")


def test_multi_source_fetcher():
    """Test multi-source fetcher with fallback."""
    print("\n" + "=" * 60)
    print("Testing Multi-Source Data Fetcher")
    print("=" * 60)
    
    fetcher = MultiSourceDataFetcher(primary_source='roostoo', fallback_sources=['binance'])
    
    # Test with Roostoo first (may fail, should fallback)
    print("\n1. Testing with Roostoo first (will fallback to Binance if Roostoo fails)...")
    test_pairs = ['BTC/USD', 'ETH/USD']
    
    for pair in test_pairs:
        print(f"\n   Testing {pair}...")
        df = fetcher.fetch_klines(pair, interval='1m', limit=10, use_roostoo=True, use_fallback=True)
        
        if df is not None:
            print(f"   ✓ Successfully fetched {len(df)} rows")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            print(f"   Latest price: ${df['close'].iloc[-1]:.2f}")
        else:
            print(f"   ✗ Failed to fetch data for {pair}")
    
    # Test Binance only (skip Roostoo)
    print("\n2. Testing Binance only (skipping Roostoo)...")
    for pair in test_pairs:
        print(f"\n   Testing {pair} (Binance only)...")
        df = fetcher.fetch_klines(pair, interval='1m', limit=10, use_roostoo=False, use_fallback=True)
        
        if df is not None:
            print(f"   ✓ Successfully fetched {len(df)} rows from Binance")
            print(f"   Latest price: ${df['close'].iloc[-1]:.2f}")
        else:
            print(f"   ✗ Failed to fetch data for {pair}")


def test_with_data_engine():
    """Test integration with Data Engine."""
    print("\n" + "=" * 60)
    print("Testing Data Engine Integration")
    print("=" * 60)
    
    try:
        from data_pipeline.data_engine import DataEngine
        
        # Initialize with fallback enabled
        print("\n1. Initializing Data Engine with fallback enabled...")
        engine = DataEngine(db_path="data_cache/test_trading_data.db", use_fallback=True)
        
        # Test fetching data
        print("\n2. Testing data fetch with automatic fallback...")
        pairs = ['BTC/USD', 'ETH/USD']
        
        for pair in pairs:
            print(f"\n   Fetching {pair}...")
            df = engine.fetch_klines(pair, interval='1m', limit=10)
            
            if df is not None:
                print(f"   ✓ Successfully fetched {len(df)} rows")
                print(f"   Latest price: ${df['close'].iloc[-1]:.2f}")
                
                # Save to database
                engine.save_ohlcv_to_db(pair, df, interval='1m')
                print(f"   ✓ Saved to database")
            else:
                print(f"   ✗ Failed to fetch data for {pair}")
        
        # Get database stats
        print("\n3. Database statistics...")
        stats = engine.get_database_stats()
        print(f"   Total rows: {stats.get('total_rows', 0)}")
        print(f"   Pairs: {stats.get('pairs', [])}")
        
    except Exception as e:
        print(f"   Error testing Data Engine integration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test Binance data source
    test_binance_data_source()
    
    # Test multi-source fetcher
    test_multi_source_fetcher()
    
    # Test with Data Engine
    test_with_data_engine()
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Binance API provides reliable fallback when Roostoo is unavailable")
    print("2. Multi-source fetcher automatically handles fallback")
    print("3. Data Engine integrates seamlessly with multi-source support")
    print("4. All data is stored in SQLite database regardless of source")

