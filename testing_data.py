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
    
    # Initialize data engine
    print("\n1. Initializing Data Engine...")
    engine = DataEngine(db_path="data_cache/trading_data.db", update_interval=60)
    
    # Get available pairs
    print("\n2. Fetching available trading pairs...")
    pairs = engine.get_available_pairs()
    print(f"   Available pairs: {pairs[:]}")

    test_pairs = pairs[:3] + ['BTC/USD']
    print(f"\n3. Testing data fetch for: {test_pairs}")

    print("\n4. Fetching OHLCV data...")
    engine.update_market_data(test_pairs, interval='1d', limit=10, use_cache=True, incremental=True)
    
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
    

if __name__ == "__main__":
    test_data_engine()