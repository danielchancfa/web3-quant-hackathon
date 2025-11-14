"""
Backtest the trading policy using Backtesting.py on historical forecasts.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import SignalStrategy, TrailingStrategy

from config import get_config
from prediction_execution import NextBarPolicyConfig, evaluate_forecast
from tools.technical_indicators import get_technical_signals


def load_forecasts(
    db_path: Path,
    pairs: List[str],
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    where = ["actual_close IS NOT NULL"]
    params: List = []

    if pairs:
        placeholders = ",".join(["?"] * len(pairs))
        where.append(f"pair IN ({placeholders})")
        params.extend(pairs)

    if start:
        where.append("as_of >= ?")
        params.append(start)
    if end:
        where.append("as_of <= ?")
        params.append(end)

    query = f"""
        SELECT pair, as_of, reference_close,
               pred_open, pred_high, pred_low, pred_close, pred_volume,
               pred_delta_close, pred_delta_open, pred_delta_high, pred_delta_low, pred_delta_volume,
               confidence,
               actual_open, actual_high, actual_low, actual_close, actual_volume
        FROM next_bar_forecasts
        WHERE {' AND '.join(where)}
        ORDER BY as_of ASC
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(query, conn, params=params)

    if df.empty:
        return df

    df["as_of"] = pd.to_datetime(df["as_of"], utc=True)
    return df


def annualized_sharpe(returns: np.ndarray, freq_per_year: float = 24 * 365) -> float:
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=1)
    if std_ret == 0:
        return 0.0
    return mean_ret / std_ret * np.sqrt(freq_per_year)


def _safe_float(value) -> float:
    """Safely convert value to float, handling binary BLOB data."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    if isinstance(value, bytes):
        # Binary BLOB data - try to decode as float32 or float64
        import struct
        try:
            if len(value) == 4:
                # Try little-endian first (most common)
                try:
                    return float(struct.unpack('<f', value)[0])
                except:
                    # Try big-endian
                    return float(struct.unpack('>f', value)[0])
            elif len(value) == 8:
                # Try little-endian first
                try:
                    return float(struct.unpack('<d', value)[0])
                except:
                    # Try big-endian
                    return float(struct.unpack('>d', value)[0])
            else:
                # Try native format
                if len(value) == 4:
                    return float(struct.unpack('f', value)[0])
                elif len(value) == 8:
                    return float(struct.unpack('d', value)[0])
        except (struct.error, ValueError, TypeError):
            # If all decoding fails, return 0.0
            return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def build_forecast_row(row: pd.Series) -> Dict[str, object]:
    pred = {
        "close": _safe_float(row.pred_close),
        "open": _safe_float(row.pred_open),
        "high": _safe_float(row.pred_high),
        "low": _safe_float(row.pred_low),
        "volume": _safe_float(row.pred_volume),
        "delta_close": _safe_float(row.pred_delta_close),
        "delta_open": _safe_float(row.pred_delta_open),
        "delta_high": _safe_float(row.pred_delta_high),
        "delta_low": _safe_float(row.pred_delta_low),
        "delta_volume": _safe_float(row.pred_delta_volume),
        "confidence": _safe_float(row.confidence),
    }
    return {
        "pair": row.pair,
        "as_of": row.as_of.isoformat(),
        "reference_close": _safe_float(row.reference_close),
        "predicted": pred,
    }


class ForecastStrategy(Strategy):
    forecasts_store: Optional[pd.DataFrame] = None
    policy_config: Optional[NextBarPolicyConfig] = None

    def init(self) -> None:
        self._index = 0
        if ForecastStrategy.forecasts_store is None or ForecastStrategy.policy_config is None:
            raise RuntimeError("ForecastStrategy requires forecasts_store and policy_config to be set before running.")
        self.forecasts = ForecastStrategy.forecasts_store.reset_index(drop=True)
        self.policy_cfg = ForecastStrategy.policy_config
        
        # Calculate moving average for trend filter if needed
        if self.policy_cfg.strategy_mode == "filter" and self.policy_cfg.use_trend_filter:
            # Calculate MA from price history
            self.ma_periods = self.policy_cfg.trend_ma_periods
        
        # For technical mode: initialize price history from data
        if self.policy_cfg.strategy_mode == "technical":
            # Initialize with available price data
            self.price_history = [float(self.data.Close[i]) for i in range(len(self.data.Close))]
            self.last_signal = None  # Track last signal to detect crossovers
            self.last_ma_fast = None
            self.last_ma_slow = None

    def next(self) -> None:
        if self._index >= len(self.forecasts):
            return

        forecast_row = self.forecasts.iloc[self._index]
        forecast = build_forecast_row(forecast_row)
        instruction = evaluate_forecast(
            forecast=forecast,
            portfolio_value=float(self.equity),
            current_position_notional=float(abs(self.position.size) * self.data.Close[-1]),
            config=self.policy_cfg,
        )

        net_price = self.data.Close[-1]
        ref_close = forecast.get("reference_close", net_price)  # Use forecast's reference price
        notional = instruction.get("notional", 0.0)
        target_price = instruction.get("target_price")
        stop_price = instruction.get("stop_price")
        
        # Technical indicator-based strategy (primary signal from indicators)
        if self.policy_cfg.strategy_mode == "technical":
            # Build price history from Backtesting.py data
            # Use all available price data up to current bar
            price_series = pd.Series([float(self.data.Close[i]) for i in range(len(self.data.Close))])
            
            # Calculate technical signals
            min_periods = max(self.policy_cfg.ma_slow_period, 26, 14)
            if len(price_series) >= min_periods:
                # Simple MA-only strategy if enabled
                if self.policy_cfg.use_simple_ma_only:
                    from tools.technical_indicators import calculate_moving_averages
                    ma_fast, ma_slow = calculate_moving_averages(
                        price_series,
                        fast=self.policy_cfg.ma_fast_period,
                        slow=self.policy_cfg.ma_slow_period
                    )
                    current_price = price_series.iloc[-1]
                    current_ma_fast = ma_fast.iloc[-1]
                    current_ma_slow = ma_slow.iloc[-1]
                    
                    tech_signals = {'overall_signal': 'hold'}
                    
                    # Only trade on actual crossovers, not just when condition is met
                    if self.last_ma_fast is not None and self.last_ma_slow is not None:
                        # Golden cross: fast MA crosses ABOVE slow MA (crossover event)
                        if self.last_ma_fast <= self.last_ma_slow and current_ma_fast > current_ma_slow:
                            if current_price > current_ma_fast:
                                tech_signals['overall_signal'] = 'buy'
                        # Death cross: fast MA crosses BELOW slow MA (crossover event)
                        elif self.last_ma_fast >= self.last_ma_slow and current_ma_fast < current_ma_slow:
                            if current_price < current_ma_fast:
                                tech_signals['overall_signal'] = 'sell'
                    
                    # Update for next iteration
                    self.last_ma_fast = current_ma_fast
                    self.last_ma_slow = current_ma_slow
                else:
                    # When requiring all 3 indicators, make thresholds more lenient
                    # Otherwise indicators rarely all signal at once
                    rsi_oversold = self.policy_cfg.rsi_oversold
                    rsi_overbought = self.policy_cfg.rsi_overbought
                    macd_threshold = self.policy_cfg.macd_signal_threshold
                    
                    if self.policy_cfg.min_indicators_agree >= 3:
                        # Make RSI more lenient: use 40/60 instead of 30/70
                        rsi_oversold = max(40.0, rsi_oversold - 10)
                        rsi_overbought = min(60.0, rsi_overbought - 10)
                        # Make MACD more lenient: lower threshold
                        macd_threshold = max(0.0, macd_threshold - 0.001)
                    
                    tech_signals = get_technical_signals(
                        price_series,
                        rsi_oversold=rsi_oversold,
                        rsi_overbought=rsi_overbought,
                        macd_threshold=macd_threshold,
                        ma_fast_period=self.policy_cfg.ma_fast_period,
                        ma_slow_period=self.policy_cfg.ma_slow_period,
                    )
                    
                    # Apply stricter filter: require all indicators to agree
                    if self.policy_cfg.min_indicators_agree >= 3:
                        buy_votes = sum([1 for k, v in tech_signals.items() if v == 'buy' and k != 'overall_signal'])
                        sell_votes = sum([1 for k, v in tech_signals.items() if v == 'sell' and k != 'overall_signal'])
                        # Require all 3 indicators to agree (all must signal same direction)
                        if buy_votes < 3:
                            tech_signals['overall_signal'] = 'hold'
                        if sell_votes < 3:
                            tech_signals['overall_signal'] = 'hold'
                
                # Use technical signal as primary, prediction as confirmation
                tech_signal = tech_signals['overall_signal']
                pred_signal = instruction.get("action", "hold")
                
                # Calculate position size based on technical signal strength
                if tech_signal == "buy":
                    # Technical says buy - use full size or reduced if prediction disagrees
                    if self.policy_cfg.use_prediction_confirmation:
                        if pred_signal == "buy":
                            # Both agree - full size
                            pass
                        elif pred_signal == "hold":
                            # Prediction neutral - reduce size by 50%
                            notional = notional * 0.5
                        else:
                            # Prediction says sell - skip trade
                            instruction["action"] = "hold"
                            instruction["reason"] = "prediction_disagrees"
                            instruction["notional"] = 0.0
                            notional = 0.0
                    else:
                        # Pure technical - use full size
                        instruction["action"] = "buy"
                        instruction["reason"] = "technical_buy"
                        # Calculate notional based on portfolio value
                        if notional == 0:
                            notional = float(self.equity) * self.policy_cfg.base_risk_fraction
                elif tech_signal == "sell":
                    # Technical says sell - close long position
                    if self.policy_cfg.use_prediction_confirmation:
                        if pred_signal == "sell":
                            # Both agree - close position
                            pass
                        elif pred_signal == "hold":
                            # Prediction neutral - reduce size
                            notional = notional * 0.5
                        else:
                            # Prediction says buy - skip
                            instruction["action"] = "hold"
                            instruction["reason"] = "prediction_disagrees"
                            instruction["notional"] = 0.0
                            notional = 0.0
                    else:
                        # Pure technical - close position
                        instruction["action"] = "sell"
                        instruction["reason"] = "technical_sell"
                else:
                    # Technical says hold - don't trade
                    instruction["action"] = "hold"
                    instruction["reason"] = "technical_hold"
                    instruction["notional"] = 0.0
                    notional = 0.0
            else:
                # Not enough data for indicators - don't trade
                instruction["action"] = "hold"
                instruction["reason"] = "insufficient_data"
                instruction["notional"] = 0.0
                notional = 0.0
        
        # Apply trend filter if in filter mode
        elif self.policy_cfg.strategy_mode == "filter" and self.policy_cfg.use_trend_filter:
            # Calculate moving average from recent price history
            if len(self.data.Close) >= self.ma_periods:
                recent_prices = [self.data.Close[i] for i in range(-self.ma_periods, 0)]
                current_ma = sum(recent_prices) / len(recent_prices)
                
                # For long trades: require price above MA AND bullish prediction
                if instruction["action"] == "buy" and net_price <= current_ma:
                    instruction["action"] = "hold"
                    instruction["reason"] = "trend_filter_failed"
                    instruction["notional"] = 0.0
                    notional = 0.0
                # For short trades: require price below MA AND bearish prediction
                elif instruction["action"] == "sell" and net_price >= current_ma:
                    instruction["action"] = "hold"
                    instruction["reason"] = "trend_filter_failed"
                    instruction["notional"] = 0.0
                    notional = 0.0
        
        # Adjust target/stop prices relative to current entry price if they're based on old reference
        # This handles cases where price moved between forecast time and execution time
        if target_price is not None and stop_price is not None:
            # If prices are based on ref_close but we're entering at net_price, adjust them
            if ref_close != net_price and ref_close > 0:
                price_ratio = net_price / ref_close
                target_price = target_price * price_ratio
                stop_price = stop_price * price_ratio

        # If force_flatten is enabled, close position at start of each bar (hourly flattening)
        if self.policy_cfg.force_flatten and self.position:
            self.position.close()

        if notional <= 0:
            self._index += 1
            return

        def _normalize_size(amount: float) -> float:
            fraction = amount / float(self.equity)
            if fraction < 1.0:
                return max(fraction, 0.0)
            return max(1.0, round(amount / net_price))

        if instruction["action"] == "buy":
            size = _normalize_size(notional)
            if size <= 0:
                self._index += 1
                return
            
            # Close opposite position if exists
            if self.position.is_short:
                self.position.close()
            
            # Use target/stop orders if provided
            # Backtesting.py requires: SL < Entry < TP for long positions
            if target_price is not None and stop_price is not None and target_price > 0 and stop_price > 0:
                # Validate prices for long position: must have SL < Entry < TP
                if stop_price < net_price < target_price:
                    # Correct order: SL < Entry < TP
                    self.buy(size=size, tp=target_price, sl=stop_price)
                elif target_price > net_price and stop_price < net_price:
                    # TP is above entry, SL is below - valid
                    self.buy(size=size, tp=target_price, sl=stop_price)
                else:
                    # Invalid price order - adjust or skip tp/sl
                    if target_price <= net_price:
                        # Target is at or below entry - prediction doesn't support long
                        # Only use stop loss if it's valid
                        if stop_price < net_price:
                            self.buy(size=size, sl=stop_price)
                        else:
                            # No valid stop either - skip tp/sl
                            self.buy(size=size)
                    elif stop_price >= net_price:
                        # Stop is at or above entry - invalid, adjust it
                        stop_price = net_price * 0.995  # 0.5% below entry
                        if target_price > net_price:
                            self.buy(size=size, tp=target_price, sl=stop_price)
                        else:
                            self.buy(size=size, sl=stop_price)
                    else:
                        # Both prices might be wrong - skip tp/sl
                        self.buy(size=size)
            else:
                self.buy(size=size)
                
        elif instruction["action"] == "sell":
            # For sell, close long position or open short
            if self.position.is_long:
                # Close long position (exit with target/stop if provided)
                if target_price is not None and stop_price is not None and target_price > 0 and stop_price > 0:
                    # For closing, we can set exit orders
                    self.position.close()
                    # Note: Backtesting.py doesn't support tp/sl on close(), 
                    # so we'll rely on the position being closed and re-entered if needed
                else:
                    self.position.close()
            elif self.policy_cfg.allow_shorts:
                # Open short position (if allowed)
                size = _normalize_size(notional)
                if size > 0:
                    if target_price is not None and stop_price is not None and target_price > 0 and stop_price > 0:
                        self.sell(size=size, tp=target_price, sl=stop_price)
                    else:
                        self.sell(size=size)

        self._index += 1


def run_backtest(
    price_df: pd.DataFrame,
    forecasts: pd.DataFrame,
    policy_cfg: NextBarPolicyConfig,
    cash: float = 1_000_000,
) -> Dict[str, object]:
    ForecastStrategy.forecasts_store = forecasts
    ForecastStrategy.policy_config = policy_cfg

    bt = Backtest(
        price_df,
        ForecastStrategy,
        cash=cash,
        commission=0.0,
        exclusive_orders=True,
    )
    stats = bt.run()
    return stats


def build_price_series(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df = df.sort_values("as_of")
    df = df.rename(
        columns={
            "actual_open": "Open",
            "actual_high": "High",
            "actual_low": "Low",
            "actual_close": "Close",
            "actual_volume": "Volume",
        }
    )
    df.index = df["as_of"]
    return df[["Open", "High", "Low", "Close", "Volume"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the trading policy using Backtesting.py.")
    parser.add_argument("--db_path", type=Path, default=None, help="Path to SQLite database.")
    parser.add_argument("--pairs", type=str, default="", help="Comma-separated list of pairs to include.")
    parser.add_argument("--start", type=str, default=None, help="Start timestamp (ISO 8601).")
    parser.add_argument("--end", type=str, default=None, help="End timestamp (ISO 8601).")
    parser.add_argument("--min_confidence", type=float, default=0.55)
    parser.add_argument("--min_edge", type=float, default=0.0005)
    parser.add_argument("--base_risk_fraction", type=float, default=0.10)
    parser.add_argument("--max_risk_fraction", type=float, default=0.20)
    parser.add_argument("--use_hourly_flatten", action="store_true", 
                       help="Use hourly flattening instead of target/stop logic")
    parser.add_argument("--strategy_mode", type=str, default="technical",
                       choices=["direct", "filter", "technical", "sizing", "hold_time"],
                       help="How to use predictions: direct, filter, technical, sizing, or hold_time")
    parser.add_argument("--no_trend_filter", action="store_false", dest="use_trend_filter",
                       help="Disable trend filter in filter mode")
    parser.set_defaults(use_trend_filter=True)
    parser.add_argument("--trend_ma_periods", type=int, default=20,
                       help="Moving average periods for trend filter")
    parser.add_argument("--fixed_stop_pct", type=float, default=None,
                       help="Fixed stop loss percentage (overrides default 0.5%%)")
    parser.add_argument("--fixed_target_pct", type=float, default=None,
                       help="Fixed target percentage (overrides default 1.25%%)")
    # Technical indicator settings
    parser.add_argument("--use_prediction_confirmation", action="store_true", default=False,
                       help="Use predictions as confirmation in technical mode")
    parser.add_argument("--rsi_oversold", type=float, default=30.0,
                       help="RSI oversold threshold")
    parser.add_argument("--rsi_overbought", type=float, default=70.0,
                       help="RSI overbought threshold")
    parser.add_argument("--macd_threshold", type=float, default=0.0,
                       help="MACD signal threshold")
    parser.add_argument("--ma_fast", type=int, default=10,
                       help="Fast moving average period")
    parser.add_argument("--ma_slow", type=int, default=20,
                       help="Slow moving average period")
    parser.add_argument("--min_indicators_agree", type=int, default=3,
                       help="Minimum indicators that must agree (2 or 3, default: 3)")
    parser.add_argument("--use_simple_ma_only", action="store_true",
                       help="Use only MA crossover strategy (simplest)")
    args = parser.parse_args()

    config = get_config()
    db_path = args.db_path or Path(config.db_path)
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()] if args.pairs else []

    data = load_forecasts(db_path=db_path, pairs=pairs, start=args.start, end=args.end)
    if data.empty:
        print(f"[WARN] No historical forecasts with actual outcomes for pairs {pairs} in the selected range.")
        return
    
    # Override stop/target if provided
    fixed_stop = args.fixed_stop_pct if args.fixed_stop_pct is not None else None
    fixed_target = args.fixed_target_pct if args.fixed_target_pct is not None else None
    
    policy_cfg = NextBarPolicyConfig(
        base_risk_fraction=args.base_risk_fraction,
        max_risk_fraction=args.max_risk_fraction,
        min_confidence=args.min_confidence,
        min_edge=args.min_edge,
        strategy_mode=args.strategy_mode,
        use_trend_filter=args.use_trend_filter,
        trend_ma_periods=args.trend_ma_periods,
        use_prediction_confirmation=args.use_prediction_confirmation,
        rsi_oversold=args.rsi_oversold,
        rsi_overbought=args.rsi_overbought,
        macd_signal_threshold=args.macd_threshold,
        ma_fast_period=args.ma_fast,
        ma_slow_period=args.ma_slow,
        min_indicators_agree=args.min_indicators_agree,
        use_simple_ma_only=args.use_simple_ma_only,
        fixed_stop_pct=fixed_stop if fixed_stop is not None else 0.005,
        fixed_target_pct=fixed_target if fixed_target is not None else 0.0125,
        allow_shorts=False,
        force_flatten=args.use_hourly_flatten,  # Allow testing with hourly flattening
    )

    price_df = build_price_series(data)
    stats = run_backtest(price_df, data, policy_cfg)

    print("=== Backtest Results ===")
    for key, value in stats.items():
        if isinstance(value, (float, np.float64)):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    equity_curve = stats["_equity_curve"]["Equity"]
    returns = equity_curve.pct_change().dropna()
    sharpe = annualized_sharpe(returns.values)
    print(f"Annualized Sharpe: {sharpe:.4f}")


if __name__ == "__main__":
    main()

