"""
Data processing utilities for transformer-based trading models.
"""

from pathlib import Path
from typing import Tuple

import sqlite3
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from config import get_config

DEFAULT_PAIR = 'BTC/USD'


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f'SELECT * FROM {table}', conn)
    return df


def load_ohlcv(db_path: Path, pair: str = DEFAULT_PAIR, interval: str = '1m') -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            '''
            SELECT pair, interval, timestamp, datetime, open, high, low, close, volume
            FROM ohlcv
            WHERE pair = ? AND interval = ?
            ORDER BY datetime
            ''',
            conn,
            params=(pair, interval),
        )
    if df.empty:
        return df
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)
    return df.reset_index(drop=True)


def _load_fear_greed_series() -> pd.DataFrame:
    """Prefer loading Fear & Greed from local CSV; fallback to DB if unavailable."""
    config = get_config()
    csv_path = getattr(config, 'fear_greed_csv_path', None)
    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            # allow relative path from project root
            alt_path = Path.cwd() / csv_path
            if alt_path.exists():
                path = alt_path
        if path.exists():
            df = pd.read_csv(path)
            if 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'], utc=True)
                df['fear_greed_value'] = pd.to_numeric(df['fng_value'], errors='coerce')
                df['fear_greed_class'] = df.get('fng_classification')
                return df[['datetime', 'fear_greed_value', 'fear_greed_class']].copy()

    # Fallback to whatever is stored in the database (possibly shorter history)
    return pd.DataFrame()


def compute_daily_features(db_path: Path, pair: str = DEFAULT_PAIR, label_mode: str = 'ternary') -> pd.DataFrame:
    ohlcv = load_ohlcv(db_path, pair, interval='1m')
    if ohlcv.empty:
        ohlcv = load_ohlcv(db_path, pair, interval='1d')
        if ohlcv.empty:
            raise ValueError(f"No OHLCV data available for {pair} (1m or 1d).")

    ohlcv['date'] = ohlcv['datetime'].dt.date
    daily = ohlcv.groupby('date').agg({
        'close': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'volume': 'sum'
    }).rename(columns={'close': 'close_price'})
    daily['return'] = daily['close_price'].pct_change().fillna(0.0)
    daily['volatility'] = (daily['high'] - daily['low']) / daily['close_price']

    tx = _load_table(db_path, 'horus_transaction_count')
    tx['datetime'] = pd.to_datetime(tx['datetime'])
    tx['date'] = tx['datetime'].dt.date
    tx = tx.groupby('date')['transaction_count'].sum()

    tvl = _load_table(db_path, 'horus_chain_tvl')
    tvl['datetime'] = pd.to_datetime(tvl['datetime'])
    tvl['date'] = tvl['datetime'].dt.date
    tvl = tvl[tvl['chain'] == 'bitcoin'].groupby('date')['tvl'].mean()

    fg = _load_fear_greed_series()
    if fg.empty:
        fg = _load_table(db_path, 'coinmarketcap_fear_greed')
        if not fg.empty:
            fg['datetime'] = pd.to_datetime(fg['datetime'])
            fg['fear_greed_value'] = pd.to_numeric(fg['fear_greed_value'], errors='coerce')
    if not fg.empty:
        fg['date'] = pd.to_datetime(fg['datetime']).dt.date
        fg = fg.groupby('date')[['fear_greed_value', 'fear_greed_class']].last()

    daily = daily.join(tx, how='left').join(tvl, how='left')
    if not fg.empty:
        daily = daily.join(fg, how='left')
    if 'fear_greed_value' not in daily.columns:
        daily['fear_greed_value'] = 50.0  # neutral baseline
    else:
        daily['fear_greed_value'] = daily['fear_greed_value'].fillna(method='ffill')
        daily['fear_greed_value'] = daily['fear_greed_value'].fillna(50.0)

    # Technical indicators derived from daily OHLCV
    for window in (7, 14, 30):
        daily[f'sma_{window}'] = daily['close_price'].rolling(window).mean()
        daily[f'ema_{window}'] = daily['close_price'].ewm(span=window, adjust=False).mean()

    # Rolling returns and volatility
    daily['return_7d'] = daily['close_price'].pct_change(periods=7)
    daily['return_30d'] = daily['close_price'].pct_change(periods=30)
    daily['volatility_30'] = daily['return'].rolling(30).std()

    # RSI (14-day)
    delta = daily['close_price'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    daily['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD (12/26 EMA and signal line 9)
    ema12 = daily['close_price'].ewm(span=12, adjust=False).mean()
    ema26 = daily['close_price'].ewm(span=26, adjust=False).mean()
    daily['macd'] = ema12 - ema26
    daily['macd_signal'] = daily['macd'].ewm(span=9, adjust=False).mean()
    daily['macd_hist'] = daily['macd'] - daily['macd_signal']

    # ATR (14-day)
    daily_high = daily['high']
    daily_low = daily['low']
    prev_close = daily['close_price'].shift(1)
    tr = pd.concat([
        (daily_high - daily_low),
        (daily_high - prev_close).abs(),
        (daily_low - prev_close).abs()
    ], axis=1).max(axis=1)
    daily['atr_14'] = tr.rolling(14).mean()

    # KDJ (stochastic oscillator)
    low_min = daily['low'].rolling(9).min()
    high_max = daily['high'].rolling(9).max()
    rsv = (daily['close_price'] - low_min) / (high_max - low_min + 1e-9) * 100
    daily['kdj_k'] = rsv.ewm(alpha=1/3).mean()
    daily['kdj_d'] = daily['kdj_k'].ewm(alpha=1/3).mean()
    daily['kdj_j'] = 3 * daily['kdj_k'] - 2 * daily['kdj_d']

    # Supertrend (period=10, multiplier=3)
    period = 10
    multiplier = 3
    hl2 = (daily['high'] + daily['low']) / 2
    atr = daily['atr_14'].fillna(method='ffill').fillna(method='bfill')
    upperband = (hl2 + multiplier * atr).fillna(method='ffill').fillna(method='bfill')
    lowerband = (hl2 - multiplier * atr).fillna(method='ffill').fillna(method='bfill')
    supertrend = pd.Series(index=daily.index, dtype=float)
    direction = pd.Series(index=daily.index, dtype=int)
    supertrend.iloc[0] = upperband.iloc[0]
    direction.iloc[0] = 1
    for i in range(1, len(daily)):
        curr_close = daily['close_price'].iloc[i]
        prev_supertrend = supertrend.iloc[i - 1]
        prev_direction = direction.iloc[i - 1]
        curr_upper = upperband.iloc[i]
        curr_lower = lowerband.iloc[i]

        if curr_close > prev_supertrend:
            direction.iloc[i] = 1
        elif curr_close < prev_supertrend:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = prev_direction

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = curr_lower if curr_lower > prev_supertrend or prev_direction == -1 else prev_supertrend
        else:
            supertrend.iloc[i] = curr_upper if curr_upper < prev_supertrend or prev_direction == 1 else prev_supertrend
    daily['supertrend'] = supertrend
    daily['supertrend_dir'] = direction

    daily = daily.replace([np.inf, -np.inf], np.nan)
    daily = daily.fillna(method='ffill').fillna(method='bfill').dropna()

    if label_mode == 'binary':
        median_ret = daily['return'].median()
        daily['regime_label'] = (daily['return'] >= median_ret).astype(int)
    else:
        quantiles = daily['return'].quantile([0.33, 0.66])

        def classify(r):
            if r <= quantiles.iloc[0]:
                return 0  # bearish
            if r >= quantiles.iloc[1]:
                return 2  # bullish
            return 1  # neutral

        daily['regime_label'] = daily['return'].apply(classify)
    daily['fear_greed_value'] = pd.to_numeric(daily['fear_greed_value'], errors='coerce').fillna(method='ffill')
    return daily


class DailyRegimeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sequence_length: int = 60, label_mode: str = 'ternary'):
        self.sequence_length = sequence_length
        feature_columns = [
            'close_price',
            'return',
            'return_7d',
            'return_30d',
            'volatility',
            'volatility_30',
            'sma_7',
            'sma_14',
            'sma_30',
            'ema_7',
            'ema_14',
            'ema_30',
            'rsi_14',
            'macd',
            'macd_signal',
            'macd_hist',
            'atr_14',
            'kdj_k',
            'kdj_d',
            'kdj_j',
            'supertrend',
            'supertrend_dir',
            'transaction_count',
            'tvl',
            'fear_greed_value',
        ]
        self.features = df[feature_columns].values
        self.labels = df['regime_label'].values.astype(int)
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0) + 1e-8
        self.features = (self.features - self.mean) / self.std
        self.label_mode = label_mode
        self.num_classes = int(self.labels.max()) + 1

    def __len__(self) -> int:
        return max(len(self.features) - self.sequence_length, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx: idx + self.sequence_length]
        y = self.labels[idx + self.sequence_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def prepare_daily_dataset(
    db_path: Path,
    sequence_length: int = 60,
    label_mode: str = 'ternary',
    pair: str = DEFAULT_PAIR,
) -> DailyRegimeDataset:
    daily = compute_daily_features(db_path, pair=pair, label_mode=label_mode)
    return DailyRegimeDataset(daily, sequence_length, label_mode)


def compute_hourly_features(db_path: Path, sequence_length: int = 24, pair: str = DEFAULT_PAIR) -> Tuple[np.ndarray, np.ndarray]:
    ohlcv_1m = load_ohlcv(db_path, pair, interval='1m')
    if not ohlcv_1m.empty:
        hourly = ohlcv_1m.set_index('datetime').resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    else:
        hourly_1h = load_ohlcv(db_path, pair, interval='1h')
        if hourly_1h.empty:
            raise ValueError(f"No hourly data available for {pair}.")
        hourly = hourly_1h.set_index('datetime')[['open', 'high', 'low', 'close', 'volume']].copy()

    hourly = hourly.sort_index()
    hourly['return'] = hourly['close'].pct_change().fillna(0.0)
    hourly['volatility'] = (hourly['high'] - hourly['low']) / hourly['close']
    hourly['volume_norm'] = (hourly['volume'] - hourly['volume'].mean()) / (hourly['volume'].std() + 1e-8)

    # Hourly technical indicators
    for window in (3, 6, 12, 24):
        hourly[f'sma_{window}'] = hourly['close'].rolling(window).mean()
        hourly[f'ema_{window}'] = hourly['close'].ewm(span=window, adjust=False).mean()

    hourly['return_6h'] = hourly['close'].pct_change(periods=6)
    hourly['return_24h'] = hourly['close'].pct_change(periods=24)
    hourly['volatility_24'] = hourly['return'].rolling(24).std()

    delta = hourly['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    hourly['rsi_14'] = 100 - (100 / (1 + rs))

    ema12 = hourly['close'].ewm(span=12, adjust=False).mean()
    ema26 = hourly['close'].ewm(span=26, adjust=False).mean()
    hourly['macd'] = ema12 - ema26
    hourly['macd_signal'] = hourly['macd'].ewm(span=9, adjust=False).mean()
    hourly['macd_hist'] = hourly['macd'] - hourly['macd_signal']

    prev_close = hourly['close'].shift(1)
    tr = pd.concat([
        (hourly['high'] - hourly['low']),
        (hourly['high'] - prev_close).abs(),
        (hourly['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    hourly['atr_14'] = tr.rolling(14).mean()

    low_min = hourly['low'].rolling(9).min()
    high_max = hourly['high'].rolling(9).max()
    rsv = (hourly['close'] - low_min) / (high_max - low_min + 1e-9) * 100
    hourly['kdj_k'] = rsv.ewm(alpha=1/3).mean()
    hourly['kdj_d'] = hourly['kdj_k'].ewm(alpha=1/3).mean()
    hourly['kdj_j'] = 3 * hourly['kdj_k'] - 2 * hourly['kdj_d']

    period = 10
    multiplier = 3
    hl2 = (hourly['high'] + hourly['low']) / 2
    atr = hourly['atr_14'].fillna(method='ffill').fillna(method='bfill')
    upperband = (hl2 + multiplier * atr).fillna(method='ffill').fillna(method='bfill')
    lowerband = (hl2 - multiplier * atr).fillna(method='ffill').fillna(method='bfill')
    supertrend = pd.Series(index=hourly.index, dtype=float)
    direction = pd.Series(index=hourly.index, dtype=int)
    supertrend.iloc[0] = upperband.iloc[0]
    direction.iloc[0] = 1
    for i in range(1, len(hourly)):
        curr_close = hourly['close'].iloc[i]
        prev_supertrend = supertrend.iloc[i - 1]
        prev_direction = direction.iloc[i - 1]
        curr_upper = upperband.iloc[i]
        curr_lower = lowerband.iloc[i]

        if curr_close > prev_supertrend:
            direction.iloc[i] = 1
        elif curr_close < prev_supertrend:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = prev_direction

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = curr_lower if curr_lower > prev_supertrend or prev_direction == -1 else prev_supertrend
        else:
            supertrend.iloc[i] = curr_upper if curr_upper < prev_supertrend or prev_direction == 1 else prev_supertrend
    hourly['supertrend'] = supertrend
    hourly['supertrend_dir'] = direction
    daily = compute_daily_features(db_path, pair)
    daily_idx = daily[['regime_label']].copy()
    daily_idx.index = pd.to_datetime(daily_idx.index)
    hourly = hourly.join(daily_idx, how='left')
    hourly['regime_label'] = hourly['regime_label'].fillna(method='ffill')
    hourly = hourly.replace([np.inf, -np.inf], np.nan)
    hourly = hourly.fillna(method='ffill').fillna(method='bfill').dropna()

    feature_columns = [
        'close',
        'return',
        'return_6h',
        'return_24h',
        'volatility',
        'volatility_24',
        'volume_norm',
        'sma_3',
        'sma_6',
        'sma_12',
        'sma_24',
        'ema_3',
        'ema_6',
        'ema_12',
        'ema_24',
        'rsi_14',
        'macd',
        'macd_signal',
        'macd_hist',
        'atr_14',
        'kdj_k',
        'kdj_d',
        'kdj_j',
        'supertrend',
        'supertrend_dir',
        'regime_label'
    ]
    features = hourly[feature_columns].values
    future_close = hourly['close'].shift(-1)
    future_return = (future_close - hourly['close']) / hourly['close']
    future_return = future_return.fillna(0.0).values
    buy_amount = np.clip(future_return, 0, None)
    sell_amount = np.clip(-future_return, 0, None)
    hold_confidence = 1 - np.clip(np.abs(future_return) / 0.01, 0, 1)
    targets = np.stack([buy_amount, sell_amount, hold_confidence], axis=1)
    return features, targets


class HourlySignalDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 24):
        self.sequence_length = sequence_length
        self.features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        self.targets = targets

    def __len__(self) -> int:
        return max(len(self.features) - self.sequence_length, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx: idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def prepare_hourly_dataset(
    db_path: Path,
    sequence_length: int = 24,
    pair: str = DEFAULT_PAIR,
) -> HourlySignalDataset:
    features, targets = compute_hourly_features(db_path, sequence_length, pair=pair)
    return HourlySignalDataset(features, targets, sequence_length)


def compute_execution_features(db_path: Path, lookback: int = 24, pair: str = DEFAULT_PAIR) -> Tuple[np.ndarray, np.ndarray]:
    ohlcv_1m = load_ohlcv(db_path, pair, interval='1m')
    if not ohlcv_1m.empty:
        hourly = ohlcv_1m.set_index('datetime').resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    else:
        hourly_1h = load_ohlcv(db_path, pair, interval='1h')
        if hourly_1h.empty:
            raise ValueError(f"No hourly data available for execution features for {pair}.")
        hourly = hourly_1h.set_index('datetime')[['open', 'high', 'low', 'close', 'volume']].copy()

    hourly = hourly.sort_index()
    hourly['return'] = hourly['close'].pct_change().fillna(0.0)
    hourly['volatility'] = (hourly['high'] - hourly['low']) / hourly['close']
    hourly['volume_norm'] = (hourly['volume'] - hourly['volume'].mean()) / (hourly['volume'].std() + 1e-8)

    features = hourly[['close', 'return', 'volatility', 'volume_norm']].values
    future_close = np.roll(hourly['close'].values, -1)
    future_return = (future_close - hourly['close'].values) / hourly['close'].values
    future_return[-1] = 0.0
    buy_amount = np.clip(future_return, 0, None)
    sell_amount = np.clip(-future_return, 0, None)
    hold_confidence = 1 - np.clip(np.abs(future_return) / 0.01, 0, 1)
    targets = np.stack([buy_amount, sell_amount, hold_confidence], axis=1)
    return features, targets


class HourlyExecutionDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, lookback: int = 24):
        self.lookback = lookback
        self.features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        self.targets = targets

    def __len__(self) -> int:
        return max(len(self.features) - self.lookback, 0)

    def __getitem__(self, idx: int):
        x = self.features[idx: idx + self.lookback]
        y = self.targets[idx + self.lookback]
        return torch.tensor(x.flatten(), dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def prepare_execution_dataset(
    db_path: Path,
    lookback: int = 24,
    pair: str = DEFAULT_PAIR,
) -> HourlyExecutionDataset:
    features, targets = compute_execution_features(db_path, lookback, pair=pair)
    return HourlyExecutionDataset(features, targets, lookback)
