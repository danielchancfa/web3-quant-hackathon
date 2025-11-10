"""
Test script for Horus Transaction Count API
--------------------------------------------
Tests the transaction_count endpoint and verifies historical data retrieval.
"""

from data_pipeline.horus_api import get_transaction_count, set_horus_api_key, get_horus_api_key
from config import get_config
import time
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_basic_transaction_count():
    """Test basic transaction count retrieval."""
    print("=" * 60)
    print("Test 1: Basic Transaction Count")
    print("=" * 60)
    
    # Check API key
    api_key = get_horus_api_key()
    if api_key:
        print(f"✓ Horus API key found")
    else:
        print("⚠ Horus API key not set. Some tests may fail.")
        print("  Set HORUS_API_KEY environment variable or add to config.json")
    
    # Test Bitcoin transaction count (daily)
    print("\n1. Testing Bitcoin transaction count (daily interval):")
    result = get_transaction_count(chain='bitcoin', interval='1d', as_dataframe=True)
    
    if result is not None and not result.empty:
        print(f"   ✓ Successfully fetched data")
        print(f"   DataFrame shape: {result.shape}")
        print(f"   Columns: {list(result.columns)}")
        print(f"   Date range: {result['datetime'].min()} -> {result['datetime'].max()}")
        print(f"   Head:\n{result.head()}")
    else:
        print("   ✗ Failed to fetch transaction count")
        return False
    
    return True


def test_historical_data():
    """Test historical data retrieval with time ranges."""
    print("\n" + "=" * 60)
    print("Test 2: Historical Data Retrieval")
    print("=" * 60)
    
    # Calculate time ranges
    end_time = int(time.time())
    
    # Test different time ranges
    test_ranges = [
        ("Last 7 days", 7 * 24 * 60 * 60),
        ("Last 30 days", 30 * 24 * 60 * 60),
        ("Last 90 days", 90 * 24 * 60 * 60),
    ]
    
    for range_name, seconds_back in test_ranges:
        print(f"\n{range_name}:")
        start_time = end_time - seconds_back
        
        print(f"   Start: {datetime.fromtimestamp(start_time)} ({start_time})")
        print(f"   End: {datetime.fromtimestamp(end_time)} ({end_time})")
        
        result = get_transaction_count(
            chain='bitcoin',
            interval='1d',
            start=start_time,
            end=end_time,
            as_dataframe=True
        )
        
        if result is not None and not result.empty:
            print(f"   ✓ Successfully fetched historical data")
            print(f"   Data points: {len(result)}")
            print(f"   First rows:\n{result.head(2)}")
            print(f"   Last rows:\n{result.tail(2)}")
        else:
            print(f"   ✗ Failed to fetch historical data")
            return False
    
    return True


def test_different_intervals():
    """Test different time intervals."""
    print("\n" + "=" * 60)
    print("Test 3: Different Time Intervals")
    print("=" * 60)
    
    intervals = ['1d']  # Add more if API supports: '1h', '1w', '1m'
    
    for interval in intervals:
        print(f"\nTesting interval: {interval}")
        result = get_transaction_count(chain='bitcoin', interval=interval, as_dataframe=True)
        
        if result is not None and not result.empty:
            print(f"   ✓ Successfully fetched data for {interval} interval")
            print(f"   Rows: {len(result)}; Date range: {result['datetime'].min()} -> {result['datetime'].max()}")
        else:
            print(f"   ✗ Failed for {interval} interval")


def test_different_chains():
    """Test different blockchain chains."""
    print("\n" + "=" * 60)
    print("Test 4: Different Blockchain Chains")
    print("=" * 60)
    
    chains = ['bitcoin']  # Add more chains as needed: 'ethereum', 'polygon', etc.
    
    for chain in chains:
        print(f"\nTesting chain: {chain}")
        result = get_transaction_count(chain=chain, interval='1d', as_dataframe=True)
        
        if result is not None and not result.empty:
            print(f"   ✓ Successfully fetched data for {chain}")
            print(f"   Rows: {len(result)}; Columns: {list(result.columns)}")
        else:
            print(f"   ✗ Failed for {chain}")


def test_data_structure_analysis():
    """Analyze the data structure returned by the API."""
    print("\n" + "=" * 60)
    print("Test 6: Data Structure Analysis")
    print("=" * 60)
    
    # Get recent data
    end_time = int(time.time())
    start_time = end_time - (7 * 24 * 60 * 60)  # Last 7 days
    
    result = get_transaction_count(
        chain='bitcoin',
        interval='1d',
        start=start_time,
        end=end_time,
        as_dataframe=True
    )
    
    if result is not None and not result.empty:
        print("\nDataFrame summary:")
        print(result.info())
        print("\nFirst rows:")
        print(result.head())
        print("\nColumn statistics:")
        print(result.describe(include='all'))
    else:
        print("No data returned to analyze.")


def main():
    """Run all tests."""
    print("Horus API Transaction Count - Comprehensive Test")
    print("=" * 60)
    
    # Check configuration
    config = get_config()
    print(f"\nConfiguration:")
    print(f"  API Key set: {'Yes' if get_horus_api_key() else 'No'}")
    print(f"  Base URL: https://api-horus.com")
    
    # Run tests
    tests = [
        ("Basic Transaction Count", test_basic_transaction_count),
        ("Historical Data", test_historical_data),
        ("Different Intervals", test_different_intervals),
        ("Different Chains", test_different_chains),
        ("Data Structure Analysis", test_data_structure_analysis),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Error in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    print("\n" + "=" * 60)
    print("Key Findings:")
    print("=" * 60)
    print("1. Check if API key is required (may work without)")
    print("2. Verify DataFrame structure and columns")
    print("3. Confirm historical data retrieval works")
    print("4. Test different chains and intervals")
    print("\nNext Steps:")
    print("- If tests pass, proceed to integrate with Data Engine")
    print("- If tests fail, check API documentation and adjust endpoint")
    print("- Analyze response structure to determine how to store data")


if __name__ == "__main__":
    main()

