# Configuration Setup Guide

This guide explains how to configure API keys and settings for the trading bot.

## Quick Start

1. **Set Environment Variables** (Recommended):
```bash
export ROOSTOO_API_KEY="your_roostoo_api_key"
export ROOSTOO_SECRET_KEY="your_roostoo_secret_key"
export HORUS_API_KEY="your_horus_api_key"
export COINMARKETCAP_API_KEY="your_coinmarketcap_api_key"
export FEAR_GREED_CSV_PATH="/path/to/CryptoGreedFear.csv"  # optional local dataset
```

2. **Or Use Config File**:
```bash
cp config.json.example config.json
# Edit config.json with your API keys
```

3. **Test Configuration**:
```bash
python config.py
```

## Configuration Methods

### Method 1: Environment Variables (Most Secure)

Set environment variables in your shell:

```bash
# Required API Keys
export ROOSTOO_API_KEY="your_roostoo_api_key"
export ROOSTOO_SECRET_KEY="your_roostoo_secret_key"
export HORUS_API_KEY="your_horus_api_key"
export COINMARKETCAP_API_KEY="your_coinmarketcap_api_key"
export FEAR_GREED_CSV_PATH="/path/to/CryptoGreedFear.csv"  # optional local dataset

# Optional Settings
export DB_PATH="data_cache/trading_data.db"
export UPDATE_INTERVAL=60
export USE_FALLBACK=true
export CACHE_MAX_SIZE=10
export LOG_LEVEL=INFO
```

**Advantages:**
- Most secure (no files with secrets)
- Easy to use in production
- Works with CI/CD systems

### Method 2: Config File

1. Copy the example file:
```bash
cp config.json.example config.json
```

2. Edit `config.json`:
```json
{
  "roostoo_api_key": "your_roostoo_api_key_here",
  "roostoo_secret_key": "your_roostoo_secret_key_here",
  "horus_api_key": "your_horus_api_key_here",
  "coinmarketcap_api_key": "your_coinmarketcap_api_key_here",
  "fear_greed_csv_path": "CryptoGreedFear.csv",
  "db_path": "data_cache/trading_data.db",
  "update_interval": 60,
  "use_fallback": true,
  "cache_max_size": 10,
  "log_level": "INFO"
}
```

**Note:** `config.json` is in `.gitignore` and won't be committed to version control.

### Method 3: Runtime Override

You can override settings when initializing:

```python
from data_pipeline.data_engine import DataEngine

# Override specific settings
engine = DataEngine(
    db_path="custom/path.db",
    update_interval=120,
    use_fallback=False,
    horus_api_key="custom_key"
)
```

## Configuration Priority

Settings are loaded in this order (higher priority overrides lower):

1. **Runtime parameters** (highest priority)
2. **Environment variables**
3. **Config file** (`config.json`)
4. **Default values** (lowest priority)

## Available Settings

### API Keys

- `ROOSTOO_API_KEY` / `roostoo_api_key`: Roostoo API key (required)
- `ROOSTOO_SECRET_KEY` / `roostoo_secret_key`: Roostoo secret key (required)
- `HORUS_API_KEY` / `horus_api_key`: Horus API key (optional, for on-chain data)
- `COINMARKETCAP_API_KEY` / `coinmarketcap_api_key`: CoinMarketCap API key (optional, for sentiment)
- `FEAR_GREED_CSV_PATH` / `fear_greed_csv_path`: Optional path to local Fear & Greed CSV

### Data Engine Settings

- `DB_PATH` / `db_path`: Database file path (default: `data_cache/trading_data.db`)
- `UPDATE_INTERVAL` / `update_interval`: Data update interval in seconds (default: `