"""
Technical indicator calculations for trading strategy.
"""
import numpy as np
import pandas as pd


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan) + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)  # Neutral RSI if not enough data


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD, signal line, and histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def calculate_moving_averages(prices: pd.Series, fast: int = 10, slow: int = 20) -> tuple:
    """Calculate fast and slow moving averages."""
    ma_fast = prices.rolling(fast).mean()
    ma_slow = prices.rolling(slow).mean()
    return ma_fast, ma_slow


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
    """Calculate Bollinger Bands."""
    ma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = ma + (std * std_dev)
    lower = ma - (std * std_dev)
    return upper, ma, lower


def get_technical_signals(
    prices: pd.Series,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    macd_threshold: float = 0.0,
    ma_fast_period: int = 10,
    ma_slow_period: int = 20,
) -> dict:
    """
    Calculate technical indicator signals.
    Returns dict with signals: 'buy', 'sell', or 'hold'
    """
    signals = {
        'rsi_signal': 'hold',
        'macd_signal': 'hold',
        'ma_signal': 'hold',
        'overall_signal': 'hold',
    }
    
    if len(prices) < max(ma_slow_period, 26, 14):
        return signals
    
    # RSI signal
    rsi = calculate_rsi(prices, period=14)
    if len(rsi) > 0:
        current_rsi = rsi.iloc[-1]
        if current_rsi < rsi_oversold:
            signals['rsi_signal'] = 'buy'
        elif current_rsi > rsi_overbought:
            signals['rsi_signal'] = 'sell'
    
    # MACD signal
    macd, macd_signal, macd_hist = calculate_macd(prices)
    if len(macd_hist) > 0:
        current_hist = macd_hist.iloc[-1]
        if current_hist > macd_threshold:
            signals['macd_signal'] = 'buy'
        elif current_hist < -macd_threshold:
            signals['macd_signal'] = 'sell'
    
    # Moving average crossover signal
    ma_fast, ma_slow = calculate_moving_averages(prices, fast=ma_fast_period, slow=ma_slow_period)
    if len(ma_fast) > 0 and len(ma_slow) > 0:
        current_price = prices.iloc[-1]
        current_ma_fast = ma_fast.iloc[-1]
        current_ma_slow = ma_slow.iloc[-1]
        
        # Golden cross: fast MA crosses above slow MA, and price above both
        if current_ma_fast > current_ma_slow and current_price > current_ma_fast:
            signals['ma_signal'] = 'buy'
        # Death cross: fast MA crosses below slow MA, and price below both
        elif current_ma_fast < current_ma_slow and current_price < current_ma_fast:
            signals['ma_signal'] = 'sell'
    
    # Overall signal: require multiple indicators to agree
    # Default: require at least 2 indicators (can be made stricter)
    buy_votes = sum([1 for k, v in signals.items() if v == 'buy' and k != 'overall_signal'])
    sell_votes = sum([1 for k, v in signals.items() if v == 'sell' and k != 'overall_signal'])
    
    # Count total indicators that gave a signal
    total_indicators = 3  # RSI, MACD, MA
    min_agreement = 2  # Require at least 2 to agree (can be made 3 for stricter)
    
    if buy_votes >= min_agreement:
        signals['overall_signal'] = 'buy'
    elif sell_votes >= min_agreement:
        signals['overall_signal'] = 'sell'
    
    return signals


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) for adaptive stop-loss/take-profit.
    
    ATR measures market volatility. Higher ATR = more volatile = wider stops needed.
    """
    # True Range = max of:
    #   1. High - Low
    #   2. abs(High - Previous Close)
    #   3. abs(Low - Previous Close)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR is the moving average of True Range
    atr = true_range.rolling(window=period).mean()
    
    return atr


def get_technical_signals_strict(
    prices: pd.Series,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    macd_threshold: float = 0.0,
    ma_fast_period: int = 10,
    ma_slow_period: int = 20,
    min_indicators_agree: int = 3,  # Require all 3 indicators to agree
) -> dict:
    """
    Stricter version: require all indicators to agree.
    """
    return get_technical_signals(
        prices, rsi_oversold, rsi_overbought, macd_threshold,
        ma_fast_period, ma_slow_period
    )

