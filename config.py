"""
Configuration Management
------------------------
Centralized configuration for API keys and settings.
Supports environment variables (recommended) and config file fallback.
"""

import os
from pathlib import Path
from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)

# Configuration file path
CONFIG_FILE = Path(__file__).parent / "config.json"
ENV_FILE = Path(__file__).parent / ".env"


class Config:
    """Configuration manager for API keys and settings."""
    
    def __init__(self):
        """Initialize configuration."""
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables and config file."""
        # First, try to load from environment variables (most secure)
        self._load_from_env()
        
        # Then, try to load from config.json file
        if CONFIG_FILE.exists():
            self._load_from_file()
        else:
            logger.info(f"Config file not found at {CONFIG_FILE}. Using environment variables only.")
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Roostoo API
        self._config['roostoo_api_key'] = os.getenv('ROOSTOO_API_KEY')
        self._config['roostoo_secret_key'] = os.getenv('ROOSTOO_SECRET_KEY')
        
        # Horus API
        self._config['horus_api_key'] = os.getenv('HORUS_API_KEY')
        
        # CoinMarketCap API
        self._config['coinmarketcap_api_key'] = os.getenv('COINMARKETCAP_API_KEY')

        # Optional local datasets
        self._config['fear_greed_csv_path'] = os.getenv('FEAR_GREED_CSV_PATH')

        # Binance API (optional, not required for public endpoints)
        self._config['binance_api_key'] = os.getenv('BINANCE_API_KEY')
        self._config['binance_secret_key'] = os.getenv('BINANCE_SECRET_KEY')
        
        # Data Engine settings
        self._config['db_path'] = os.getenv('DB_PATH', 'data_cache/trading_data.db')
        self._config['update_interval'] = int(os.getenv('UPDATE_INTERVAL', '60'))
        self._config['use_fallback'] = os.getenv('USE_FALLBACK', 'true').lower() == 'true'
        self._config['cache_max_size'] = int(os.getenv('CACHE_MAX_SIZE', '10'))
        self._config['fetch_intraday_market_data'] = os.getenv('FETCH_INTRADAY_MARKET_DATA', 'true').lower() == 'true'
        
        # Logging
        self._config['log_level'] = os.getenv('LOG_LEVEL', 'INFO')
    
    def _load_from_file(self):
        """Load configuration from config.json file."""
        try:
            with open(CONFIG_FILE, 'r') as f:
                file_config = json.load(f)
                
                # Merge file config with env config (env takes precedence)
                for key, value in file_config.items():
                    if key not in self._config or self._config[key] is None:
                        self._config[key] = value
                        
            logger.info(f"Loaded configuration from {CONFIG_FILE}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config file {CONFIG_FILE}: {e}")
        except Exception as e:
            logger.error(f"Error loading config file {CONFIG_FILE}: {e}")
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: str):
        """Set configuration value (runtime only, not persisted)."""
        self._config[key] = value
    
    def save_to_file(self):
        """Save current configuration to config.json file."""
        try:
            # Create a copy without sensitive defaults
            save_config = {
                'db_path': self._config.get('db_path', 'data_cache/trading_data.db'),
                'update_interval': self._config.get('update_interval', 60),
                'use_fallback': self._config.get('use_fallback', True),
                'cache_max_size': self._config.get('cache_max_size', 10),
                'fetch_intraday_market_data': self._config.get('fetch_intraday_market_data', True),
                'log_level': self._config.get('log_level', 'INFO'),
            }
            
            # Only save API keys if they're set (user should use env vars instead)
            if self._config.get('roostoo_api_key'):
                save_config['roostoo_api_key'] = self._config['roostoo_api_key']
            if self._config.get('roostoo_secret_key'):
                save_config['roostoo_secret_key'] = self._config['roostoo_secret_key']
            if self._config.get('horus_api_key'):
                save_config['horus_api_key'] = self._config['horus_api_key']
            if self._config.get('coinmarketcap_api_key'):
                save_config['coinmarketcap_api_key'] = self._config['coinmarketcap_api_key']
            if self._config.get('fear_greed_csv_path'):
                save_config['fear_greed_csv_path'] = self._config['fear_greed_csv_path']
            
            with open(CONFIG_FILE, 'w') as f:
                json.dump(save_config, f, indent=2)
            
            logger.info(f"Saved configuration to {CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
    
    # Property accessors for convenience
    @property
    def roostoo_api_key(self) -> Optional[str]:
        """Get Roostoo API key."""
        return self._config.get('roostoo_api_key')
    
    @property
    def roostoo_secret_key(self) -> Optional[str]:
        """Get Roostoo secret key."""
        return self._config.get('roostoo_secret_key')
    
    @property
    def horus_api_key(self) -> Optional[str]:
        """Get Horus API key."""
        return self._config.get('horus_api_key')
    
    @property
    def coinmarketcap_api_key(self) -> Optional[str]:
        """Get CoinMarketCap API key."""
        return self._config.get('coinmarketcap_api_key')

    @property
    def fear_greed_csv_path(self) -> Optional[str]:
        """Get custom path for Fear & Greed CSV."""
        return self._config.get('fear_greed_csv_path')

    @property
    def binance_api_key(self) -> Optional[str]:
        """Get Binance API key (optional)."""
        return self._config.get('binance_api_key')
    
    @property
    def binance_secret_key(self) -> Optional[str]:
        """Get Binance secret key (optional)."""
        return self._config.get('binance_secret_key')
    
    @property
    def db_path(self) -> str:
        """Get database path."""
        return self._config.get('db_path', 'data_cache/trading_data.db')
    
    @property
    def update_interval(self) -> int:
        """Get update interval in seconds."""
        return self._config.get('update_interval', 60)
    
    @property
    def use_fallback(self) -> bool:
        """Get use fallback setting."""
        return self._config.get('use_fallback', True)
    
    @property
    def cache_max_size(self) -> int:
        """Get cache max size."""
        return self._config.get('cache_max_size', 10)
    
    @property
    def log_level(self) -> str:
        """Get log level."""
        return self._config.get('log_level', 'INFO')

    @property
    def fetch_intraday_market_data(self) -> bool:
        """Get flag for fetching intraday market data from Roostoo/Binance."""
        return self._config.get('fetch_intraday_market_data', True)


# Global config instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reload_config():
    """Reload configuration."""
    global _config_instance
    _config_instance = Config()
    return _config_instance


# Convenience functions
def get_roostoo_api_key() -> Optional[str]:
    """Get Roostoo API key."""
    return get_config().roostoo_api_key


def get_roostoo_secret_key() -> Optional[str]:
    """Get Roostoo secret key."""
    return get_config().roostoo_secret_key


def get_horus_api_key() -> Optional[str]:
    """Get Horus API key."""
    return get_config().horus_api_key


def get_coinmarketcap_api_key() -> Optional[str]:
    """Get CoinMarketCap API key."""
    return get_config().coinmarketcap_api_key


def get_fear_greed_csv_path() -> Optional[str]:
    """Get custom path for Fear & Greed CSV."""
    return get_config().fear_greed_csv_path


def get_binance_api_key() -> Optional[str]:
    """Get Binance API key."""
    return get_config().binance_api_key


def get_binance_secret_key() -> Optional[str]:
    """Get Binance secret key."""
    return get_config().binance_secret_key


# Example usage
if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    
    print("Configuration loaded:")
    print(f"  Roostoo API Key: {'***' if config.roostoo_api_key else 'Not set'}")
    print(f"  Roostoo Secret Key: {'***' if config.roostoo_secret_key else 'Not set'}")
    print(f"  Horus API Key: {'***' if config.horus_api_key else 'Not set'}")
    print(f"  CoinMarketCap API Key: {'***' if config.coinmarketcap_api_key else 'Not set'}")
    print(f"  Fear & Greed CSV Path: {config.fear_greed_csv_path or 'Not set'}")
    print(f"  Database Path: {config.db_path}")
    print(f"  Update Interval: {config.update_interval}s")
    print(f"  Use Fallback: {config.use_fallback}")
    print(f"  Log Level: {config.log_level}")
    
    # Create example config file if it doesn't exist
    if not CONFIG_FILE.exists():
        print(f"\nCreating example config file at {CONFIG_FILE}")
        example_config = {
            "db_path": "data_cache/trading_data.db",
            "update_interval": 60,
            "use_fallback": True,
            "cache_max_size": 10,
            "log_level": "INFO",
            "roostoo_api_key": "your_roostoo_api_key_here",
            "roostoo_secret_key": "your_roostoo_secret_key_here",
            "horus_api_key": "your_horus_api_key_here",
            "coinmarketcap_api_key": "your_coinmarketcap_api_key_here",
            "fear_greed_csv_path": "CryptoGreedFear.csv"
        }
        config.save_to_file()
        print("Example config file created. Please update with your API keys.")
        print("Note: For security, prefer using environment variables instead.")

