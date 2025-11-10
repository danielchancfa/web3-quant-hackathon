"""
Test script for On-Chain Data Integration
------------------------------------------
Tests Horus API integration and on-chain data features in the Data Engine.
"""

from data_pipeline.data_engine import DataEngine
from data_pipeline.horus_api import (
    get_transaction_count,
    get_chain_tvl,
    get_whale_net_flow,
    get_mining_work,
    set_horus_api_key,
    get_horus_api_key
)
from config import get_config
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_horus_api():
    """Test Horus API directly."""
    print("=" * 60)
    print("Testing Horus API")
    print("=" * 60)
    
    api_key = get_horus_api_key()
    if api_key:
        print("✓ Horus API key found in config")
    else:
        print("⚠ Horus API key not found. Some tests may fail.")
        print("  Set HORUS_API_KEY environment variable or add to config.json")
    
    print("\n1. Testing transaction count (bitcoin)...")
    tx_df = get_transaction_count('bitcoin', interval='1d', as_dataframe=True)
    if tx_df is not None and not tx_df.empty:
        print(f"   ✓ Rows: {len(tx_df)}, Columns: {list(tx_df.columns)}")
        print(f"   Latest value: {tx_df['transaction_count'].iloc[-1]}")
    else:
        print("   ✗ Failed to fetch transaction count")
    
    print("\n2. Testing chain TVL (ethereum)...")
    tvl_df = get_chain_tvl('ethereum', interval='1d', as_dataframe=True)
    if tvl_df is not None and not tvl_df.empty:
        print(f"   ✓ Rows: {len(tvl_df)}, Latest TVL: {tvl_df['tvl'].iloc[-1]}")
    else:
        print("   ✗ Failed to fetch TVL")
    
    print("\n3. Testing whale net flow (bitcoin)...")
    whale_df = get_whale_net_flow('bitcoin', interval='1d', as_dataframe=True)
    if whale_df is not None and not whale_df.empty:
        print(f"   ✓ Rows: {len(whale_df)}, Latest flow: {whale_df['whale_net_flow'].iloc[-1]}")
    else:
        print("   ✗ Failed to fetch whale net flow")
    
    print("\n4. Testing mining work (bitcoin)...")
    mining_df = get_mining_work('bitcoin', interval='1d', as_dataframe=True)
    if mining_df is not None and not mining_df.empty:
        print(f"   ✓ Rows: {len(mining_df)}, Latest work: {mining_df['mining_work'].iloc[-1]}")
    else:
        print("   ⚠ Mining work not available (only for Bitcoin)")


def test_data_engine_onchain():
    """Test on-chain data integration in Data Engine."""
    print("\n" + "=" * 60)
    print("Testing Data Engine On-Chain Integration")
    print("=" * 60)
    
    engine = DataEngine(db_path="data_cache/test_trading_data.db")
    
    print("\n1. Testing on-chain data fetch...")
    token_network_pairs = [('ETH', 'ethereum'), ('BTC', 'bitcoin')]
    
    for token, network in token_network_pairs:
        print(f"\n   Fetching on-chain data for {token} ({network})...")
        onchain_data = engine.fetch_onchain_data(token, network=network)
        
        if onchain_data:
            print(f"   ✓ Keys: {list(onchain_data.keys())}")
            engine.save_onchain_data_to_db(onchain_data)
            print(f"   ✓ Saved summary metrics to database")
        else:
            print(f"   ✗ Failed to fetch on-chain data for {token}")
    
    print("\n2. Testing on-chain data loading from database...")
    for token, network in token_network_pairs:
        onchain_df = engine.load_onchain_data_from_db(token, network=network)
        if onchain_df is not None and not onchain_df.empty:
            print(f"   ✓ Loaded {len(onchain_df)} rows for {token}")
            print(f"   Columns: {list(onchain_df.columns)}")
        else:
            print(f"   ⚠ No on-chain data in database for {token}")
    
    print("\n3. Testing exchange flows...")
    flows = engine.fetch_exchange_flows('binance', 'ETH', network='ethereum')
    if flows is not None and not flows.empty:
        print(f"   ✓ Successfully fetched exchange flows")
        print(f"   Rows: {len(flows)}")
        print(f"   Columns: {list(flows.columns)}")
    else:
        print("   ⚠ No exchange flows data (may require API key or different endpoint)")
    
    print("\n4. Testing combined features (market + on-chain)...")
    
    # First, ensure we have market data
    print("   Fetching market data for ETH/USD...")
    engine.update_market_data(['ETH/USD'], interval='1m', limit=100, incremental=True)
    
    # Get combined features
    combined = engine.get_combined_features(
        'ETH/USD',
        token_symbol='ETH',
        interval='1m',
        lookback=50,
        network='ethereum',
        include_onchain=True
    )
    
    if combined is not None and not combined.empty:
        print(f"   ✓ Successfully combined market and on-chain data")
        print(f"   Total rows: {len(combined)}")
        print(f"   Total columns: {len(combined.columns)}")
        
        # Check for on-chain columns
        onchain_cols = [col for col in combined.columns if 'onchain' in col.lower() or col.startswith('onchain_')]
        if onchain_cols:
            print(f"   On-chain columns: {onchain_cols}")
        else:
            print("   ⚠ No on-chain columns found (may need to fetch on-chain data first)")
        
        print(f"\n   Sample data:")
        print(combined.tail(3).to_string())
    else:
        print("   ⚠ Failed to get combined features")
    
    # Database statistics
    print("\n5. Database statistics...")
    stats = engine.get_database_stats()
    print(f"   Total OHLCV rows: {stats.get('total_rows', 0)}")
    
    # Check on-chain data count
    try:
        with engine._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM onchain_data')
            result = cursor.fetchone()
            onchain_count = result['count'] if result else 0
            print(f"   Total on-chain data rows: {onchain_count}")
            
            cursor.execute('SELECT COUNT(*) as count FROM exchange_flows')
            result = cursor.fetchone()
            flows_count = result['count'] if result else 0
            print(f"   Total exchange flows rows: {flows_count}")
    except Exception as e:
        print(f"   Error getting statistics: {e}")


def test_continuous_onchain_collection():
    """Test continuous collection with on-chain data."""
    print("\n" + "=" * 60)
    print("Testing Continuous On-Chain Data Collection")
    print("=" * 60)
    
    engine = DataEngine(
        db_path="data_cache/test_trading_data.db",
        update_interval=300  # 5 minutes
    )
    
    print("\nThis would start continuous collection...")
    print("(Not running in test mode to avoid long waits)")
    print("\nTo run continuous collection:")
    print("  engine.start_continuous_collection(['ETH/USD'], interval='1m')")
    print("  # Then separately fetch on-chain data periodically:")
    print("  engine.fetch_onchain_data('ETH', network='ethereum')")
    print("  engine.save_onchain_data_to_db(onchain_data)")


if __name__ == "__main__":
    test_horus_api()
    
    test_data_engine_onchain()
    
    test_continuous_onchain_collection()
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Horus API provides on-chain data for trading signals")
    print("2. On-chain data is stored in SQLite database")
    print("3. Combined features merge market data with on-chain metrics")
    print("4. Exchange flows show inflow/outflow patterns")
    print("\nNext Steps:")
    print("1. Set HORUS_API_KEY environment variable or add to config.json")
    print("2. Fetch on-chain data periodically (e.g., hourly or daily)")
    print("3. Use get_combined_features() for model training/inference")
    print("4. Monitor exchange flows for market sentiment signals")
    print("\nConfiguration:")
    print("  - API keys can be set via environment variables (recommended)")
    print("  - Or via config.json file (see config.json.example)")
    print("  - Run 'python config.py' to test configuration loading")

