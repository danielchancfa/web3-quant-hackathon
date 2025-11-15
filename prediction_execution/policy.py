"""
Next-bar forecast policy.

Translates predicted close + confidence into actionable trade instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Any


@dataclass
class NextBarPolicyConfig:
    base_risk_fraction: float = 0.10  # baseline 10% of portfolio per trade
    max_risk_fraction: float = 0.20   # cap per pair (20%)
    min_confidence: float = 0.55
    min_edge: float = 5e-4            # ~5 bps move required
    # Strategy mode: how to use predictions
    strategy_mode: str = "hybrid"  # "direct", "filter", "technical", "sizing", "hold_time", "hybrid"
    # For "hybrid" mode: combine prediction model and MA strategy
    hybrid_prediction_weight: float = 0.5  # 50% allocation to prediction model
    hybrid_ma_weight: float = 0.5          # 50% allocation to MA strategy
    # For "filter" mode: require trend confirmation
    use_trend_filter: bool = True      # Require price above/below MA for entry
    trend_ma_periods: int = 20        # Moving average period for trend filter
    # For "technical" mode: technical indicator settings
    use_prediction_confirmation: bool = False  # Use predictions as weak confirmation in technical mode
    rsi_oversold: float = 30.0        # RSI oversold threshold
    rsi_overbought: float = 70.0      # RSI overbought threshold
    macd_signal_threshold: float = 0.0  # MACD signal threshold
    ma_fast_period: int = 10          # Fast MA period
    ma_slow_period: int = 20          # Slow MA period
    min_indicators_agree: int = 3     # Require all 3 indicators to agree (stricter)
    use_simple_ma_only: bool = True   # Use only MA crossover (simplest strategy) - DEFAULT FOR COMPETITION
    # Target/stop settings - use fixed percentages for simplicity and reliability
    use_fixed_stops: bool = True      # Use fixed % stops instead of prediction-based
    fixed_stop_pct: float = 0.01      # Fixed stop loss: 1% of entry price (competition setting)
    fixed_target_pct: float = 0.02    # Fixed target: 2% of entry price (1:2 ratio, optimized for 45% win rate)
    # Prediction-based settings (if use_fixed_stops=False)
    target_buffer: float = 0.05      # small buffer above predicted_close (5% of range)
    stop_buffer: float = 0.15         # buffer below predicted_low (15% of range)
    risk_reward_ratio: float = 2.5   # minimum target/stop ratio (2.5:1)
    min_stop_distance_pct: float = 0.005  # minimum stop distance (0.5% of entry price)
    allow_shorts: bool = False
    force_flatten: bool = False       # use target/stop logic instead of hourly flattening
    exit_after_seconds: int = 3600


def _parse_datetime(ts_str: str) -> datetime:
    try:
        # Accept both naive and timezone-aware ISO strings
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def evaluate_forecast(
    forecast: Dict[str, Any],
    portfolio_value: float,
    current_position_notional: float,
    config: NextBarPolicyConfig | None = None,
) -> Dict[str, Any]:
    """
    Convert a single forecast entry into an instruction dict.
    """
    cfg = config or NextBarPolicyConfig()
    predicted = forecast["predicted"]
    ref_close = float(forecast["reference_close"])
    confidence = float(predicted.get("confidence", 0.0))
    pred_close = float(predicted.get("close", ref_close))
    pred_open = float(predicted.get("open", ref_close))
    pred_high = float(predicted.get("high", pred_close))
    pred_low = float(predicted.get("low", pred_close))
    edge = float(predicted.get("delta_close", (pred_close - ref_close) / ref_close if ref_close > 0 else 0.0))
    range_abs = max(pred_high - pred_low, 1e-6 * ref_close)
    pred_volume = float(predicted.get("volume", 0.0))

    as_of = _parse_datetime(forecast["as_of"])
    exit_time = as_of + timedelta(seconds=cfg.exit_after_seconds)

    instruction: Dict[str, Any] = {
        "pair": forecast["pair"],
        "action": "hold",
        "notional": 0.0,
        "confidence": confidence,
        "edge": edge,
        "target_price": None,
        "stop_price": None,
        "exit_time": exit_time.isoformat(),
        "predicted_volume": pred_volume,
        "predicted_open": pred_open,
        "predicted_high": pred_high,
        "predicted_low": pred_low,
        "predicted_close": pred_close,
        "reason": "",
    }

    if confidence < cfg.min_confidence:
        instruction["reason"] = "low_confidence"
        # Only flatten if force_flatten is enabled
        if current_position_notional > 0 and cfg.force_flatten:
            instruction.update({"action": "sell", "notional": current_position_notional, "reason": "flatten_low_conf"})
        return instruction

    if abs(edge) < cfg.min_edge:
        instruction["reason"] = "edge_too_small"
        # Only flatten if force_flatten is enabled
        if current_position_notional > 0 and cfg.force_flatten:
            instruction.update({"action": "sell", "notional": current_position_notional, "reason": "flatten_no_edge"})
        return instruction

    # Strategy mode: how to use predictions
    # NOTE: Technical indicators (MA crossover) require price history
    # For live trading, this should be handled by the executor with price history from database/API
    # For now, we pass through predictions - executor can override based on technical signals
    if cfg.strategy_mode == "hybrid":
        # Hybrid mode: combine prediction model and MA strategy
        # Allocate percentage to prediction-based trades
        # MA signals will be checked separately and combined
        instruction["strategy_mode"] = "hybrid"
        instruction["prediction_weight"] = cfg.hybrid_prediction_weight
        instruction["ma_weight"] = cfg.hybrid_ma_weight
        # Continue with prediction-based calculation, but weight will be applied later
    elif cfg.strategy_mode == "filter":
        # Filter mode: use predictions as confirmation, not direct signal
        # Need trend information from price history - will be checked in backtest
        # For now, just pass through the prediction signal
        # The backtest will add trend filter
        pass
    elif cfg.strategy_mode == "sizing":
        # Sizing mode: use predictions to size positions, always trade
        # This will be handled differently - always allow trade
        pass
    elif cfg.strategy_mode == "hold_time":
        # Hold time mode: use predictions to determine exit time
        # This will be handled in exit logic
        pass
    # "direct" mode: use predictions directly (default behavior)

    max_position = portfolio_value * cfg.max_risk_fraction
    # Scale base risk by confidence, but ensure minimum effective risk
    base_notional = portfolio_value * cfg.base_risk_fraction * confidence
    
    # For hybrid mode, allocate only the prediction weight portion
    if cfg.strategy_mode == "hybrid":
        base_notional = base_notional * cfg.hybrid_prediction_weight
        instruction["prediction_notional"] = base_notional
        instruction["requires_ma_signal"] = True  # Flag that MA signal should also be checked
    
    # If we already have a position, only add to it if we're below max
    desired_notional = min(base_notional, max(max_position - current_position_notional, 0.0))
    desired_notional = max(desired_notional, 0.0)

    if edge > 0:
        if desired_notional <= 0:
            instruction["reason"] = "no_capacity"
            return instruction
        instruction["action"] = "buy"
        instruction["notional"] = desired_notional
        instruction["reason"] = "long_forecast"
        
        if cfg.use_fixed_stops:
            # Simple fixed percentage stops - more reliable
            stop_price = ref_close * (1 - cfg.fixed_stop_pct)
            target_price = ref_close * (1 + cfg.fixed_target_pct)
        else:
            # Prediction-based stops (original logic)
            # Target: predicted close + small buffer (our prediction is the target)
            target_price = pred_close + cfg.target_buffer * range_abs
            
            # Stop: predicted low - small buffer (protect against worst case)
            stop_price = max(0.0, pred_low - cfg.stop_buffer * range_abs)
            
            # Ensure minimum risk/reward ratio, but don't make stops too tight
            # Calculate potential gain (target - entry) and loss (entry - stop)
            potential_gain = target_price - ref_close
            potential_loss = ref_close - stop_price
            
            if potential_loss > 0:
                actual_ratio = potential_gain / potential_loss
                if actual_ratio < cfg.risk_reward_ratio:
                    # Adjust stop to meet minimum ratio (tighten stop)
                    # target_gain / stop_loss >= ratio => stop_loss <= target_gain / ratio
                    max_stop_loss = potential_gain / cfg.risk_reward_ratio
                    # But enforce minimum stop distance to avoid stops that are too tight
                    min_stop_loss = ref_close * cfg.min_stop_distance_pct
                    max_stop_loss = max(max_stop_loss, min_stop_loss)
                    stop_price = ref_close - max_stop_loss
                    stop_price = max(0.0, stop_price)
        
        instruction["target_price"] = target_price
        instruction["stop_price"] = stop_price
        return instruction

    # Negative edge
    if current_position_notional > 0:
        instruction["action"] = "sell"
        instruction["notional"] = min(current_position_notional, current_position_notional + desired_notional)
        instruction["reason"] = "exit_on_short_signal"
        
        # For closing long: target is predicted close (lower), stop is predicted high (higher)
        target_price = pred_close - cfg.target_buffer * range_abs
        stop_price = pred_high + cfg.stop_buffer * range_abs
        
        # Ensure minimum risk/reward ratio (for exit, we want to limit loss)
        potential_gain = ref_close - target_price  # gain from selling at target
        potential_loss = stop_price - ref_close     # loss if price goes to stop
        
        if potential_loss > 0:
            actual_ratio = potential_gain / potential_loss
            if actual_ratio < cfg.risk_reward_ratio:
                max_stop_loss = potential_gain / cfg.risk_reward_ratio
                stop_price = ref_close + max_stop_loss
        
        instruction["target_price"] = target_price
        instruction["stop_price"] = stop_price
        return instruction

    if cfg.allow_shorts and desired_notional > 0:
        instruction["action"] = "sell"
        instruction["notional"] = desired_notional
        instruction["reason"] = "open_short"
        
        # For short: target is predicted close (lower), stop is predicted high (higher)
        target_price = pred_close - cfg.target_buffer * range_abs
        stop_price = pred_high + cfg.stop_buffer * range_abs
        
        # Ensure minimum risk/reward ratio
        potential_gain = ref_close - target_price  # gain from short at target
        potential_loss = stop_price - ref_close     # loss if price goes to stop
        
        if potential_loss > 0:
            actual_ratio = potential_gain / potential_loss
            if actual_ratio < cfg.risk_reward_ratio:
                max_stop_loss = potential_gain / cfg.risk_reward_ratio
                stop_price = ref_close + max_stop_loss
        
        instruction["target_price"] = target_price
        instruction["stop_price"] = stop_price
        return instruction

    instruction["reason"] = "negative_edge_no_position"
    return instruction

