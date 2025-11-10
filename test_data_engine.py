"""
Test script for Data Engine
----------------------------
Quick test to verify data fetching and processing works correctly.
"""

from data_pipeline.data_engine import DataEngine
import logging

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_data_engine():
    """Test the Data Engine functionality."""
    
    print("=" * 60)
    print("Testing Data Engine")
    print("=" * 60)
    
    # Initialize data engine (uses config for API keys)
    print("\n1. Initializing Data Engine...")
    engine = DataEngine()
    
    # Get available pairs
    print("\n2. Fetching available trading pairs...")
    pairs = engine.get_available_pairs()
    print(f"   Available pairs: {pairs[:10] if len(pairs) > 10 else pairs}")
    
    if not pairs:
        print("   Warning: No pairs found, using default pairs")
        pairs = ['BTC/USD', 'ETH/USD']
    
    # Test with first few pairs
    test_pairs = pairs[:3] if len(pairs) >= 3 else pairs
    print(f"\n3. Testing data fetch for: {test_pairs}")
    
    # Fetch data
    print("\n4. Fetching OHLCV data...")
    engine.update_market_data(test_pairs, interval='1m', limit=100, use_cache=True, incremental=True)
    
    # Check results
    print("\n5. Checking fetched data...")
    for pair in test_pairs:
        # Query from database
        df = engine.query_data(pair, interval='1m', limit=100)
        if df is not None and len(df) > 0:
            print(f"\n   {pair}:")
            print(f"   - Rows in DB: {len(df)}")
            print(f"   - Date range: {df.index[0]} to {df.index[-1]}")
            print(f"   - Latest price: ${df['close'].iloc[-1]:.2f}")
            print(f"   - Latest volume: {df['volume'].iloc[-1]:.2f}")
            print(f"   - Columns: {list(df.columns)}")
        else:
            print(f"   {pair}: No data available")
    
    # Get database statistics
    print("\n6. Getting database statistics...")
    stats = engine.get_database_stats()
    print(f"   - Total rows: {stats.get('total_rows', 0)}")
    print(f"   - Pairs in DB: {stats.get('pairs', [])}")
    print(f"   - Intervals: {stats.get('intervals', [])}")
    if 'date_range' in stats:
        print(f"   - Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    
    # Test feature engineering
    print("\n7. Testing feature engineering...")
    for pair in test_pairs:
        features = engine.get_latest_features(
            pair,
            interval='1m',
            lookback=50, 
            include_indicators=True,
            normalize=False
        )
        if features is not None and len(features) > 0:
            print(f"\n   {pair} - Features with indicators:")
            print(f"   - Rows: {len(features)}")
            print(f"   - Columns: {len(features.columns)}")
            indicator_cols = [c for c in features.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            print(f"   - Indicator columns: {indicator_cols[:10]}")
            if 'rsi' in features.columns:
                print(f"   - Latest RSI: {features['rsi'].iloc[-1]:.2f}")
            if 'macd' in features.columns:
                print(f"   - Latest MACD: {features['macd'].iloc[-1]:.4f}")
    
    # Test data querying
    print("\n8. Testing data querying from database...")
    if test_pairs:
        pair = test_pairs[0]
        # Query last 100 rows
        df = engine.query_data(pair, interval='1m', limit=100)
        if df is not None and len(df) > 0:
            print(f"   {pair} - Queried from database:")
            print(f"   - Rows: {len(df)}")
            print(f"   - Date range: {df.index[0]} to {df.index[-1]}")
    
    # Test data resampling
    print("\n9. Testing data resampling...")
    if test_pairs:
        pair = test_pairs[0]
        df = engine.query_data(pair, interval='1m', limit=500)
        if df is not None and len(df) > 60:  # Need enough data to resample
            resampled = engine.resample_data(df, '5m')
            print(f"   {pair} - Resampled to 5m:")
            print(f"   - Original rows: {len(df)}")
            print(f"   - Resampled rows: {len(resampled)}")
            print(f"   - Resampled columns: {list(resampled.columns)}")
    
    print("\n" + "=" * 60)
    print("Data Engine Test Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Verify the data looks correct")
    print("2. Check the SQLite database at data_cache/trading_data.db")
    print("3. Use engine.start_continuous_collection() for live data collection")
    print("4. Use engine.query_data() for flexible data queries")
    print("5. Integrate with your Signal Engine for trading signals")
    print("\nTip: You can inspect the database using:")
    print("   sqlite3 data_cache/trading_data.db")
    print("   .tables")
    print("   SELECT * FROM ohlcv LIMIT 10;")

if __name__ == "__main__":
    test_data_engine()

