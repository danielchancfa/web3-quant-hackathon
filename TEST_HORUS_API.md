# Testing Horus API Transaction Count Endpoint

This guide helps you test the Horus API transaction count endpoint and verify historical data retrieval.

## Quick Test

### 1. Install Dependencies

```bash
pip install requests
```

### 2. Set API Key (if required)

```bash
export HORUS_API_KEY="your_api_key_here"
```

Or add to `config.json`:
```json
{
  "horus_api_key": "your_api_key_here"
}
```

### 3. Run Test Script

```bash
python test_horus_transaction_count.py
```

## Manual Testing

### Test 1: Basic Request (DataFrame)

```python
from data_pipeline.horus_api import get_transaction_count

# Basic request - Bitcoin, daily interval
df = get_transaction_count(chain='bitcoin', interval='1d', as_dataframe=True)
print(df.head())
```

**Expected Response:**
- If successful: pandas DataFrame with columns `datetime`, `timestamp`, `chain`, `interval`, `transaction_count`
- If failed: `None` (check logs for error details)

### Test 2: Historical Data (Last 7 Days)

```python
import time
from data_pipeline.horus_api import get_transaction_count

# Calculate time range
end_time = int(time.time())
start_time = end_time - (7 * 24 * 60 * 60)  # 7 days ago

# Get historical data
df = get_transaction_count(
    chain='bitcoin',
    interval='1d',
    start=start_time,
    end=end_time,
    as_dataframe=True
)
print(df.tail())
```

**Expected Response:**
- Time-series DataFrame with transaction counts for each day

### Test 3: Different Time Ranges

```python
import time
from data_pipeline.horus_api import get_transaction_count

end_time = int(time.time())

# Test different ranges
ranges = {
    "Last 7 days": 7 * 24 * 60 * 60,
    "Last 30 days": 30 * 24 * 60 * 60,
    "Last 90 days": 90 * 24 * 60 * 60,
}

for name, seconds in ranges.items():
    start_time = end_time - seconds
    df = get_transaction_count(
        chain='bitcoin',
        interval='1d',
        start=start_time,
        end=end_time,
        as_dataframe=True
    )
    print(f"{name}: {df is not None and not df.empty}")
```

## Understanding the Response

The API returns JSON that is normalized into a pandas DataFrame. Common structures:

### Example 1: List of Objects
```json
[
  {"timestamp": 1234567890, "count": 250000},
  {"timestamp": 1234654290, "count": 255000},
  ...
]
```

### Example 2: Nested Structure
```json
{
  "data": [
    {"timestamp": 1234567890, "count": 250000},
    ...
  ],
  "metadata": {...}
}
```

### Format 3: Simple Dictionary
```json
{
  "2024-01-01": 250000,
  "2024-01-02": 255000,
  ...
}
```

## Troubleshooting

### Issue: API returns 401 Unauthorized
**Solution**: Set your Horus API key
```bash
export HORUS_API_KEY="your_key"
```

### Issue: API returns 404 Not Found
**Solution**: 
- Check endpoint URL: `https://api-horus.com/blockchain/transaction_count`
- Verify chain name is correct (e.g., 'bitcoin', not 'btc')
- Check API documentation for correct endpoint path

### Issue: No historical data returned
**Solution**:
- Verify `start` and `end` parameters are in seconds (Unix timestamp)
- Check if the time range is valid (not too far in the past/future)
- Ensure `interval` parameter is correct

### Issue: Response format unexpected
**Solution**:
- Check actual API response structure
- Adjust parsing logic in `horus_api.py` if needed

## Chain TVL Tests

### Fetch All Chains as DataFrame

```python
from data_pipeline.horus_api import get_chain_tvl

chains = ['ethereum', 'solana', 'tron', 'bitcoin', 'bsc', 'base']
df = get_chain_tvl(chains, interval='1d', as_dataframe=True)
print(df.head())
print(df['chain'].unique())
```

### Fetch Single Chain

```python
df = get_chain_tvl('ethereum', interval='1d', as_dataframe=True)
print(df.tail())
```

## Whale Net Flow Tests

```python
from data_pipeline.horus_api import get_whale_net_flow

# Whale net flow for Bitcoin (default interval 1d)
df_whale = get_whale_net_flow('bitcoin', interval='1d', as_dataframe=True)
print(df_whale.head())
```

## Next Steps

Once you verify the endpoint works:

1. **Analyze Response Structure**: Understand the exact format returned
2. **Update Data Engine**: Integrate transaction count data into Data Engine
3. **Add Storage**: Store historical transaction counts in SQLite database
4. **Create Features**: Use transaction count as a feature for trading signals

## Example: Full Test

```python
from data_pipeline.horus_api import get_transaction_count
import time
from datetime import datetime

# Test basic request
print("1. Basic request:")
df = get_transaction_count(chain='bitcoin', interval='1d', as_dataframe=True)
print(f"   Success: {df is not None and not df.empty}")
if df is not None and not df.empty:
    print(f"   Shape: {df.shape}")
    print(f"   Head:\n{df.head()}")

# Test historical data
print("\n2. Historical data (last 7 days):")
end_time = int(time.time())
start_time = end_time - (7 * 24 * 60 * 60)
df = get_transaction_count(
    chain='bitcoin',
    interval='1d',
    start=start_time,
    end=end_time,
    as_dataframe=True
)
print(f"   Success: {df is not None and not df.empty}")
if df is not None and not df.empty:
    print(f"   Data points: {len(df)}")
    print(f"   Tail:\n{df.tail()}")
```

