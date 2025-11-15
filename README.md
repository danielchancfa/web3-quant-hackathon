# AI Web3 Trading Bot - Hybrid Transformer & Technical Analysis Strategy

A sophisticated algorithmic trading system combining deep learning transformer models with classical technical analysis for the Roostoo Trading Competition.

## 🎯 Strategy Overview

Our trading strategy employs a **hybrid approach** that combines the predictive power of transformer-based deep learning models with the reliability of proven technical indicators. This dual-model architecture provides robust signal generation while maintaining risk-adjusted returns across diverse market conditions.

### Core Philosophy

- **Diversification of Signals**: Combine multiple independent signal sources to reduce false positives
- **Adaptive Risk Management**: Dynamic position sizing and stop-loss/take-profit based on market volatility (ATR)
- **Data-Driven Decisions**: Continuous data pipeline ensures models operate on fresh, high-quality market data
- **Robust Execution**: Self-contained scheduler with integrated data collection, inference, and trade execution

## 🌐 Trading Universe

### Pair Selection Strategy

Our trading strategy focuses on the **top 20 market capitalization cryptocurrencies** (excluding stablecoins) for several critical reasons:

#### Why Top Market Cap Pairs?

1. **Liquidity Advantage**:
   - Higher trading volumes ensure better execution prices
   - Reduced slippage and market impact
   - More stable bid-ask spreads
   - Lower transaction costs

2. **Data Quality & History**:
   - Longer price history provides more training data
   - More reliable technical patterns and indicators
   - Better model generalization (less overfitting)
   - Richer feature sets from historical data

3. **Market Efficiency**:
   - More efficient price discovery
   - Less prone to manipulation
   - Better correlation with broader market trends
   - More predictable volatility patterns

4. **Model Performance**:
   - Transformer models require substantial historical data
   - Top market cap pairs have sufficient data for robust training
   - Better signal-to-noise ratio in price movements
   - More reliable prediction accuracy

#### Current Trading Pairs

We actively trade **13 pairs** from the top market cap universe:
- **Major Cryptocurrencies**: BTC/USD, ETH/USD, BNB/USD, SOL/USD
- **Altcoins**: ADA/USD, LINK/USD, DOGE/USD, LTC/USD, XRP/USD
- **Emerging Assets**: SUI/USD, TRX/USD, XLM/USD, ZEC/USD

#### Exclusion Criteria

- **Stablecoins**: Excluded due to minimal price volatility (USDT, USDC, DAI, etc.)
- **Low Liquidity Pairs**: Pairs with insufficient trading volume or data history
- **New Listings**: Pairs without sufficient historical data for model training

This focused approach ensures:
- ✅ Higher quality signals from well-trained models
- ✅ Better execution in liquid markets
- ✅ Reduced risk from illiquid or volatile pairs
- ✅ More consistent performance across market conditions

## 🧠 Hybrid Model Architecture

### 1. Prediction Model (Transformer-Based Deep Learning)

Our prediction model uses a **three-layer transformer architecture** trained on historical price data, on-chain metrics, and sentiment indicators:

#### Model Stack:
- **Daily Regime Transformer**: Classifies market regime (bullish/bearish/neutral) using 60-day sequences
- **Hourly Signal Transformer**: Predicts short-term price movements using 36-hour sequences  
- **Execution Model**: Generates buy/sell/hold signals with confidence scores

#### Training Process:
1. **Data Collection**: Multi-source data pipeline fetches:
   - OHLCV price data (1h, 1d intervals)
   - On-chain metrics (transaction counts, TVL, whale flows)
   - Sentiment data (Fear & Greed Index)
   
2. **Feature Engineering**: 
   - Technical indicators (RSI, MACD, Bollinger Bands, ATR)
   - Price returns and volatility measures
   - Normalized features for model input

3. **Training**: 
   - Separate models per trading pair (13 pairs: ADA, BTC, ETH, BNB, LINK, SOL, DOGE, LTC, SUI, TRX, XLM, XRP, ZEC)
   - Trained on historical data with proper train/validation splits
   - Models saved to `model_checkpoints/{PAIR}/`

#### Inference:
- Models run inference every hour on fresh data
- Outputs: predicted price movements, confidence scores, and directional signals
- Confidence threshold: 0.55 minimum for trade execution

### 2. Technical Analysis Model (Moving Average Crossover)

A complementary **classical technical analysis** component:

- **Fast MA**: 10-period moving average
- **Slow MA**: 20-period moving average
- **Signals**:
  - **Golden Cross** (Buy): Fast MA crosses above Slow MA + price > Fast MA
  - **Death Cross** (Sell): Fast MA crosses below Slow MA + price < Fast MA
- **Crossover Detection**: Tracks previous MA values to detect actual crossovers (not just conditions)

### 3. Hybrid Signal Combination

The two models are combined with **equal weighting (50/50)**:

- **Both Agree**: Full position size (100% allocation)
- **One Signals**: Half position size (50% allocation)  
- **Both Hold**: No trade

This approach:
- ✅ Reduces false signals (requires agreement)
- ✅ Captures opportunities when one model is confident
- ✅ Maintains diversification across signal types

## 💰 Position Sizing & Risk Management

### Position Sizing Logic

- **Base Risk**: 10% of portfolio value per trade
- **Confidence Scaling**: Position size scales with model confidence
  - Formula: `notional = portfolio_value × 0.10 × confidence`
- **Maximum Position**: 20% of portfolio per trading pair (prevents over-concentration)
- **Hybrid Allocation**: 
  - Prediction model: 50% of base risk
  - MA strategy: 50% of base risk
  - Combined when both agree: 100% of base risk

### Risk Management

#### Adaptive Stop-Loss & Take-Profit (ATR-Based)

We use **Average True Range (ATR)** to adapt to market volatility:

- **Stop-Loss**: Entry price - (ATR × 1.5)
- **Take-Profit**: Entry price + (ATR × 2.5)
- **Safety Limits**: 
  - Minimum stop: 0.5% of entry price
  - Maximum stop: 5% of entry price
- **Risk/Reward Ratio**: ~1.67:1 (optimized for 45%+ win rate)

#### Position Monitoring

- **Continuous Monitoring**: Every hour, all open positions are checked
- **Automatic Exits**: Positions close automatically when:
  - Stop-loss is hit (limit losses)
  - Take-profit is hit (lock in gains)
  - Model generates SELL signal (prediction or MA death cross)

## 📊 Buy/Sell Decision Logic

### Entry Signals

1. **Prediction Model Entry**:
   - Confidence ≥ 0.55
   - Predicted price movement ≥ 5 basis points
   - Positive edge (predicted close > reference close for BUY)

2. **MA Strategy Entry**:
   - Golden cross detected (Fast MA crosses above Slow MA)
   - Current price > Fast MA (confirms trend)

3. **Hybrid Entry**:
   - Both models agree → Full position
   - Only one model signals → Half position

### Exit Signals

1. **Automatic Exits** (Priority):
   - Stop-loss hit (ATR-based)
   - Take-profit hit (ATR-based)

2. **Model-Based Exits**:
   - Prediction model: Negative signal (predicted price drop)
   - MA strategy: Death cross (Fast MA crosses below Slow MA)
   - Both models agree to SELL → Close 100% of position
   - One model says SELL → Close 50% of position

## 🔄 Trading Workflow

### Hourly Execution Cycle

1. **Data Pipeline Update** (Step 1):
   - Fetches fresh price data from Roostoo API
   - Updates on-chain metrics (Horus API)
   - Updates sentiment data (CoinMarketCap)
   - Stores in SQLite database (`horus_prices_1h` table)

2. **Position Refresh** (Step 2):
   - Fetches current balances from exchange
   - Updates portfolio valuation
   - Checks all open positions

3. **Inference** (Step 3):
   - Runs prediction models for all pairs (batch processing)
   - Calculates MA signals for all pairs
   - Generates combined hybrid signals

4. **Trade Execution** (Step 4):
   - For each pair:
     - Check stop-loss/take-profit (close if hit)
     - Evaluate entry signals (prediction + MA)
     - Combine signals with weights
     - Execute trades via Roostoo API
   - Update position manager

5. **Sleep** (Step 5):
   - Wait for loop interval (default: 3600 seconds = 1 hour)
   - Repeat cycle

### Auto-Detection & Pair Management

- **Automatic Pair Discovery**: 
  - Scans `model_checkpoints/` for available models
  - Queries database for pairs with price data
  - Uses intersection (pairs both models can trade)
- **Open Position Monitoring**: 
  - Always monitors pairs with open positions
  - Ensures old positions (e.g., from previous runs) can be closed
  - Prevents orphaned positions

## 🏗️ System Architecture

The project follows a modular architecture:

1. **Data Engine** (`data_pipeline/`) - Collects, preprocesses, and stores market data
2. **Prediction Engine** (`prediction_model/`, `inference_service.py`) - Deep learning inference
3. **Policy Engine** (`prediction_execution/policy.py`) - Signal generation and position sizing
4. **Execution Engine** (`execution/`) - Trade execution via Roostoo API
5. **Scheduler** (`scheduler.py`) - Orchestrates the entire trading loop

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

