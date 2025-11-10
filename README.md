# AI Web3 Trading Bot - Data Engine

This is the Data Engine component of the AI Web3 Trading Bot for the Roostoo Trading Competition.

## 🏗️ Architecture

The project follows a modular architecture:

1. **Data Engine** (`data_engine.py`) - Collects, preprocesses, and feeds structured data
2. **Signal Engine** (TODO) - Generates trading signals using AI/ML models
3. **Execution Engine** (TODO) - Executes trades via Roostoo API

## 📦 Setup

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

#### Option 1: Environment Variables (Recommended - Most Secure)

```bash
# Set environment variables
export ROOSTOO_API_KEY="your_roostoo_api_key"
export ROOSTOO_SECRET_KEY="your_roostoo_secret_key"
export HORUS_API_KEY="your_horus_api_key"
export COINMARKETCAP_API_KEY="your_coinmarketcap_api_key"
export FEAR_GREED_CSV_PATH="/path/to/CryptoGreedFear.csv"  # optional local dataset

# Optional: Data Engine settings
export DB_PATH="data_cache/trading_data.db"
export UPDATE_INTERVAL=60
export USE_FALLBACK=true
```

#### Option 2: Config File

1. Copy the example config file:
```bash
cp config.json.example config.json
```

2. Edit `config.json` and add your API keys:
```json
{
  "roostoo_api_key": "your_roostoo_api_key_here",
  "roostoo_secret_key": "your_roostoo_secret_key_here",
  "horus_api_key": "your_horus_api_key_here",
  "coinmarketcap_api_key": "your_coinmarketcap_api_key_here",
  "fear_greed_csv_path": "CryptoGreedFear.csv",
  "db_path": "data_cache/trading_data.db",
  "update_interval": 60,
  "use_fallback": true
}
```

**Note**: The `config.json` file is in `.gitignore` and will not be committed to version control.

#### Option 3: Test Configuration

Test your configuration:
```bash
python config.py
```

This will show which API keys are configured and help verify your setup.

## 🚀 Usage

### Quick Start - Test Data Engine

```bash
python test_data_engine.py
```

This will:
- Fetch available trading pairs from Roostoo
- Download 1-minute OHLCV data for the first few pairs
- Add technical indicators
- Cache data locally

### Using Data Engine in Your Code

```python
from data_pipeline.data_engine import DataEngine

# Initialize (API keys loaded from config automatically)
engine = DataEngine()  # Uses settings from config

# Or override specific settings
engine = DataEngine(
    db_path="data_cache/trading_data.db",
    update_interval=60,
    use_fallback=True
)

# Get available pairs
pairs = engine.get_available_pairs()

# Update market data (incremental updates)
engine.update_market_data(['BTC/USD', 'ETH/USD'], interval='1m', limit=1000, incremental=True)

# Get database statistics
stats = engine.get_database_stats()
print(f"Total rows: {stats['total_rows']}")
print(f"Pairs: {stats['pairs']}")

# Query data from database
df = engine.query_data('BTC/USD', interval='1m', limit=100)

# Get latest features with technical indicators
features = engine.get_latest_features('BTC/USD', interval='1m', lookback=100, include_indicators=True)
print(features.tail())

# Start continuous data collection
engine.start_continuous_collection(['BTC/USD', 'ETH/USD'], interval='1m', run_forever=True)
```

## 📊 Data Engine Features

### 1. Market Data Collection

- **OHLCV Data**: Fetches 1-minute (or other intervals) candlestick data
- **Multiple Pairs**: Supports fetching data for multiple trading pairs
- **Multi-Source Support**: Automatically falls back to Binance API when Roostoo is unavailable or rate-limited
- **SQLite Database**: Efficiently stores all data in SQLite for fast querying
- **Incremental Updates**: Only fetches new data since last update
- **Continuous Updates**: Can run continuously to keep data up-to-date
- **Flexible Queries**: Query data by pair, interval, date range, or limit
- **Automatic Fallback**: Seamlessly switches to alternative data sources when primary API fails

### 2. Feature Engineering

The Data Engine automatically adds technical indicators:

- **Returns**: Price returns and log returns
- **Moving Averages**: SMA and EMA (5, 10, 20, 50, 100, 200 periods)
- **Volatility**: Rolling volatility (20-period, annualized)
- **RSI**: Relative Strength Index (14-period)
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Upper, lower, middle bands, width, position
- **ATR**: Average True Range (14-period)
- **Volume Indicators**: Volume moving average and volume ratio
- **Price Position**: Relative position within high-low range

### 3. Data Processing

- **Normalization**: Min-max, z-score, or robust normalization
- **Resampling**: Convert data to different time intervals (5m, 1h, 1d, etc.)
- **Data Cleaning**: Automatic deduplication and NaN handling

### 4. Data Storage

- **SQLite Database**: All data stored in `data_cache/trading_data.db`
- **Efficient Indexing**: Indexed on pair, interval, and timestamp for fast queries
- **Metadata Tracking**: Tracks last update time and data statistics in database
- **Memory Cache**: Keeps recently accessed data in memory for faster access
- **Query Interface**: Flexible SQL queries for date ranges, limits, and filtering

### 5. Database Benefits

- **Better Performance**: SQLite is more efficient than pickle files for large datasets
- **Query Flexibility**: Use SQL to query specific date ranges, pairs, etc.
- **Concurrent Access**: Supports concurrent reads (multiple processes can read simultaneously)
- **Data Integrity**: ACID transactions ensure data consistency
- **Scalability**: Can handle millions of rows efficiently
- **Inspectable**: Can inspect data using standard SQLite tools

## ⛓️ On-Chain Data Integration

The Data Engine integrates with Horus API to fetch on-chain blockchain data:

- **Transaction Count** (`get_transaction_count`): Daily transaction counts per chain (e.g., Bitcoin)
- **Mining Work** (`get_mining_work`): Average mining work per block in zettahashes (ZH)
- **Chain TVL** (`get_chain_tvl`): Chain total value locked (TVL) in USD for major networks
- **Whale Net Flow** (`get_whale_net_flow`): Net change in whale balances (native token) per chain
# Fetch whale net flow (Bitcoin example)
from data_pipeline.horus_api import get_whale_net_flow
whale_flow_df = get_whale_net_flow('bitcoin', interval='1d', as_dataframe=True)

### Usage

```python
from data_pipeline.data_engine import DataEngine
import os

# Set Horus API key (or use environment variable)
engine = DataEngine(horus_api_key=os.getenv('HORUS_API_KEY'))

# Fetch on-chain data
onchain_data = engine.fetch_onchain_data('ETH', network='ethereum')
engine.save_onchain_data_to_db(onchain_data)

# Get combined features (market + on-chain)
features = engine.get_combined_features(
    'ETH/USD',
    token_symbol='ETH',
    interval='1m',
    lookback=100,
    network='ethereum',
    include_onchain=True
)

# Fetch exchange flows
flows = engine.fetch_exchange_flows('binance', 'ETH', network='ethereum')

# Fetch chain TVL for multiple chains (Horus API helper)
from data_pipeline.horus_api import get_chain_tvl
tvl_df = get_chain_tvl(['ethereum', 'solana', 'tron', 'bitcoin', 'bsc', 'base'], interval='1d', as_dataframe=True)

# Fetch CoinMarketCap Fear & Greed index (DataFrame)
from data_pipeline.coinmarketcap_api import get_fear_and_greed_historical
fg_df = get_fear_and_greed_historical(as_dataframe=True)
```

## 🧠 Sentiment Data (CoinMarketCap)

- **Fear & Greed Index** (`get_fear_and_greed_historical`): Historical sentiment scores (0–100) with classifications
- Designed for transformer/ML pipelines: returns normalized pandas DataFrame
- Requires `COINMARKETCAP_API_KEY`

### Setup

1. Get Horus API key from [Horus Data](https://horusdata.xyz)
2. Set environment variable: `export HORUS_API_KEY=your_api_key`
3. Or add to `config.json`: `"horus_api_key": "your_api_key"`
4. Or pass directly: `DataEngine(horus_api_key='your_api_key')` (runtime override)

### Database Schema

On-chain data is stored in:
- `onchain_data`: Network and token metrics
- `exchange_flows`: Exchange inflow/outflow data

## 🔄 Multi-Source Data Support

The Data Engine now supports multiple data sources with automatic fallback:

### Primary Source: Roostoo API
- Tries Roostoo API first for all data requests
- Falls back automatically if rate-limited (429 error) or unavailable

### Fallback Source: Binance Public API
- **Free**: No API key required for public market data
- **Reliable**: High uptime and fast response times
- **Comprehensive**: Supports all major trading pairs
- **Automatic Mapping**: Converts Roostoo pairs (e.g., "BTC/USD") to Binance symbols (e.g., "BTCUSDT")

### Usage

```python
from data_pipeline.data_engine import DataEngine

# Initialize with fallback enabled (default)
engine = DataEngine(use_fallback=True)  # Will use Binance if Roostoo fails

# Or disable fallback to use only Roostoo
engine = DataEngine(use_fallback=False)

# The Data Engine automatically handles fallback
engine.update_market_data(['BTC/USD'], interval='1m', limit=1000)
```

### Pair Mapping

The system automatically maps Roostoo pairs to Binance symbols:
- `BTC/USD` → `BTCUSDT`
- `ETH/USD` → `ETHUSDT`
- `SOL/USD` → `SOLUSDT`
- And many more...

## 🔌 API Endpoints

The `data_pipeline/roostoo_api.py` module provides:

### Public Endpoints
- `check_server_time()` - Get server time
- `get_exchange_info()` - Get available trading pairs
- `get_ticker(pair)` - Get current ticker data
- `get_klines(pair, interval, limit)` - Get OHLCV/kline data ⭐

### Signed Endpoints (Require Authentication)
- `get_balance()` - Get wallet balances
- `get_pending_count()` - Get pending order count
- `place_order()` - Place buy/sell orders
- `query_order()` - Query order history
- `cancel_order()` - Cancel orders

## 📁 Project Structure

```
.
├── data_pipeline/
│   ├── roostoo_api.py              # Roostoo API client
│   ├── horus_api.py                # Horus API client (on-chain data)
│   ├── coinmarketcap_api.py        # CoinMarketCap sentiment client
│   ├── data_sources.py             # Multi-source data fetcher (Binance fallback)
│   ├── data_engine.py              # Data Engine module (SQLite-based)
│   └── historical_data_pipeline.py # End-to-end historical data extractor
├── modeling/
│   ├── data_processing.py          # Feature engineering for models
│   ├── transformer_trading.py      # Model definitions
│   └── train_transformer.py        # Training entry point
├── config.py                   # Configuration management
├── config.json.example         # Example configuration file
├── .gitignore                  # Git ignore file (excludes config.json)
├── test_data_engine.py         # Test script for Data Engine
├── test_data_sources.py        # Test script for multi-source data
├── test_onchain_data.py        # Test script for on-chain data
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── data_cache/                 # Data directory (created automatically)
    └── trading_data.db         # SQLite database with OHLCV, on-chain data
```

## 🧪 Testing

Run the test scripts to verify everything works:

```bash
# Test Data Engine with SQLite
python test_data_engine.py

# Test multi-source data fetching (Roostoo + Binance fallback)
python test_data_sources.py

# Test on-chain data integration (Horus API)
python test_onchain_data.py

# Test Horus metrics + whale net flow quick checks
python quick_test_horus.py

# Test CoinMarketCap Fear & Greed endpoint
python quick_test_coinmarketcap.py

# Test CoinMarketCap OHLCV helper (requires API key)
python test_coinmarketcap_ohlcv.py

# Build full historical dataset (market + on-chain + sentiment)
python -m data_pipeline.historical_data_pipeline
```

Expected output:
- List of available trading pairs
- Successfully fetched OHLCV data (Roostoo minutes, CoinMarketCap daily ~2-year history & hourly lookback for top-20 market-cap pairs on Roostoo)
- Automatic fallback to Binance when Roostoo fails
- Technical indicators calculated
- Data stored in SQLite database

## ⚙️ Configuration Management

The project uses a centralized configuration system (`config.py`) that supports:

1. **Environment Variables** (Recommended): Most secure, doesn't require files
2. **Config File** (`config.json`): Convenient for development
3. **Runtime Override**: Pass parameters directly to functions/classes

### Configuration Priority

1. Runtime parameters (highest priority)
2. Environment variables
3. Config file (`config.json`)
4. Default values (lowest priority)

### Available Configuration Options

- `ROOSTOO_API_KEY` / `roostoo_api_key`: Roostoo API key
- `ROOSTOO_SECRET_KEY` / `roostoo_secret_key`: Roostoo secret key
- `HORUS_API_KEY` / `horus_api_key`: Horus API key
- `COINMARKETCAP_API_KEY` / `coinmarketcap_api_key`: CoinMarketCap API key
- `DB_PATH` / `db_path`: Database file path
- `UPDATE_INTERVAL` / `update_interval`: Data update interval (seconds)
- `USE_FALLBACK` / `use_fallback`: Enable Binance fallback (true/false)
- `CACHE_MAX_SIZE` / `cache_max_size`: Maximum in-memory cache size
- `LOG_LEVEL` / `log_level`: Logging level (INFO, DEBUG, etc.)

### Testing Configuration

```bash
python config.py
```

This will display your current configuration (without showing API keys in plain text).

## ⚠️ Important Notes

1. **API Keys Security**: Never commit API keys to version control. Use environment variables or ensure `config.json` is in `.gitignore`.

2. **API Rate Limits**: The competition rules prohibit high-frequency trading. Be mindful of API request frequency.

3. **Data Cache**: The Data Engine caches data locally. Clear the `data_cache/` directory if you need fresh data.

4. **Error Handling**: The Data Engine handles API errors gracefully and falls back to cached data when available.

5. **Endpoint Format**: The `get_klines()` endpoint may return data in different formats. The Data Engine handles multiple response formats automatically.

6. **Configuration Files**: The `config.json` file is excluded from version control via `.gitignore`. Always use `config.json.example` as a template.

## 🔜 Next Steps

1. **Test API Endpoints**: Verify that `get_klines()` works with your API credentials
2. **Customize Indicators**: Add or modify technical indicators as needed
3. **Integrate Signal Engine**: Connect the Data Engine to your trading signal generator
4. **Add Sentiment Data**: Integrate Horus API or other sentiment data sources
5. **Add On-Chain Data**: Integrate blockchain metrics if available
6. **Integrate Fear & Greed**: Use CoinMarketCap sentiment data in your models

## 📚 References

- [Roostoo API Documentation](https://github.com/roostoo/Roostoo-API-Documents)
- [Horus Data Partner](https://horusdata.xyz)
- [CoinMarketCap API Documentation](https://coinmarketcap.com/api/)
- Competition Rules: See competition details for constraints and requirements

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'requests'`
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Issue: API returns 401 Unauthorized
**Solution**: 
- Check your API keys are set correctly
- Verify environment variables: `echo $ROOSTOO_API_KEY`
- Or check config.json file
- Run `python config.py` to test configuration

### Issue: No data returned from API
**Solution**: 
- Check internet connection
- Verify API endpoint is correct
- Check API response format (may need to adjust parsing in `fetch_klines()`)

### Issue: `get_klines()` returns None or empty data
**Solution**: 
- The endpoint format might be different. Check the actual API response format
- You may need to adjust the endpoint URL or parameters
- Try using `get_ticker()` first to verify API connectivity
- **With fallback enabled**: The Data Engine will automatically use Binance API if Roostoo fails

### Issue: Roostoo returns 429 (Rate Limited)
**Solution**: 
- The Data Engine automatically falls back to Binance when it detects 429 errors
- Ensure `use_fallback=True` when initializing DataEngine (this is the default)
- The system will seamlessly switch to Binance without manual intervention

### Issue: Database locked errors
**Solution**: 
- SQLite supports concurrent reads but only one write at a time
- If running multiple processes, ensure they're only reading, not writing simultaneously
- Consider using a connection pool for multiple writers

### Issue: Inspecting the database
**Solution**: 
```bash
sqlite3 data_cache/trading_data.db
.tables
SELECT * FROM ohlcv LIMIT 10;
SELECT pair, COUNT(*) FROM ohlcv GROUP BY pair;
SELECT * FROM onchain_data LIMIT 10;
SELECT * FROM exchange_flows LIMIT 10;
```

### Issue: Horus API returns 401 Unauthorized
**Solution**: 
- Set your Horus API key: `export HORUS_API_KEY=your_api_key`
- Or add to config.json: `"horus_api_key": "your_api_key"`
- Or pass it directly: `DataEngine(horus_api_key='your_api_key')` (runtime override)
- Get API key from [Horus Data](https://horusdata.xyz)
- Run `python config.py` to verify configuration

### Issue: On-chain data not available
**Solution**: 
- Check if Horus API key is set correctly
- Verify the token symbol and network are correct
- Some tokens may not have on-chain data available
- Check API response format - may need to adjust parsing in `horus_api.py`

## 📝 License

This project is for the Roostoo Trading Competition and must be open-source.

## Model Training

1. Run the data pipeline to ensure the SQLite database is populated:
   ```bash
   python -m data_pipeline.historical_data_pipeline
   ```
2. Install training dependencies (PyTorch is included in `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```
3. Train the three-layer stack locally (adjust epochs/batch sizes as needed):
   ```bash
   python -m modeling.train_transformer --epochs_daily 25 --epochs_hourly 30 --epochs_execution 15 --seq_daily 60 --daily_label_mode ternary
   ```
   Models are saved under `model_checkpoints/` by default:
   - `daily_regime_transformer.pt`
   - `hourly_signal_transformer.pt`
   - `hourly_execution_model.pt`
   *Set `--daily_label_mode binary` to train a risk-on vs risk-off classifier instead of the ternary regime model.*
4. To train on a remote GPU server, copy the project or the relevant scripts/database and run the same command over SSH. You can specify a custom output path:
   ```bash
   python -m modeling.train_transformer --output_dir /path/to/checkpoints
   ```

> **Note:** The current feature engineering and labels are baseline heuristics. Refine the feature set (e.g., richer technical indicators, alternative quantile thresholds, custom loss functions) to better match your trading strategy.

