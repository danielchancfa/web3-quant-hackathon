"""
Scheduler that ties inference, policy, and execution together.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from config import get_config
from inference_service import InferenceService
from execution.policy import apply_policy, PolicyConfig
from execution.executor import execute_trade, ExecutionConfig, get_last_price
from execution.position_manager import PositionManager
from tools.technical_indicators import calculate_moving_averages
import sqlite3
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trading loop.")
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--pairs', type=str, required=True,
                        help="Comma-separated pairs, e.g., BTC/USD,ETH/USD")
    parser.add_argument('--seq_daily', type=int, default=60)
    parser.add_argument('--seq_hourly', type=int, default=36)
    parser.add_argument('--seq_execution', type=int, default=36)
    parser.add_argument('--daily_label_mode', type=str, default='binary', choices=['binary', 'ternary'])
    parser.add_argument('--loop_interval', type=int, default=3600, help="Seconds between inference cycles")
    parser.add_argument('--position_limit', type=float, default=0.3,
                        help="Max fraction of portfolio in any single asset")
    parser.add_argument('--paper', action='store_true', help="Dry run without sending orders")
    parser.add_argument('--hybrid', action='store_true', help="Use hybrid strategy (prediction + MA)")
    parser.add_argument('--hybrid_prediction_weight', type=float, default=0.5,
                        help="Weight for prediction model in hybrid mode (default: 0.5)")
    parser.add_argument('--hybrid_ma_weight', type=float, default=0.5,
                        help="Weight for MA strategy in hybrid mode (default: 0.5)")
    parser.add_argument('--ma_fast', type=int, default=10, help="Fast MA period for hybrid mode")
    parser.add_argument('--ma_slow', type=int, default=20, help="Slow MA period for hybrid mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = [p.strip() for p in args.pairs.split(',') if p.strip()]
    config = get_config()
    db_path = Path(config.db_path)
    checkpoint_dir = Path(args.checkpoint_dir)

    service = InferenceService(
        db_path=db_path,
        checkpoint_dir=checkpoint_dir,
        seq_daily=args.seq_daily,
        seq_hourly=args.seq_hourly,
        seq_execution=args.seq_execution,
        daily_label_mode=args.daily_label_mode,
        pairs=pairs,
    )

    policy_config = PolicyConfig()
    execution_config = ExecutionConfig()
    position_manager = PositionManager(pairs)
    
    # Hybrid mode: MA strategy settings
    use_hybrid = args.hybrid if hasattr(args, 'hybrid') else False
    hybrid_prediction_weight = args.hybrid_prediction_weight if hasattr(args, 'hybrid_prediction_weight') else 0.5
    hybrid_ma_weight = args.hybrid_ma_weight if hasattr(args, 'hybrid_ma_weight') else 0.5
    ma_fast_period = args.ma_fast if hasattr(args, 'ma_fast') else 10
    ma_slow_period = args.ma_slow if hasattr(args, 'ma_slow') else 20
    
    # Store MA state for crossover detection
    ma_state = {pair: {'last_ma_fast': None, 'last_ma_slow': None} for pair in pairs}

    while True:
        position_manager.refresh()
        total_value = position_manager.total_value()
        logger.info(f"Cash USD: {position_manager.cash_usd:.2f}")
        if position_manager.asset_qty:
            logger.info("Asset quantities:")
            for asset, qty in position_manager.asset_qty.items():
                logger.info(f"  {asset}: {qty:.6f}")
        if position_manager.pair_notionals:
            logger.info("Pair notionals:")
            for pair, notional in position_manager.pair_notionals.items():
                logger.info(f"  {pair}: ${notional:.2f}")

        results = service.run_once()
        logger.info(json.dumps(results, indent=2))
        logger.info(f"Portfolio value: {total_value:.2f}")

        for pair in pairs:
            current_position = position_manager.get_pair_notional(pair)
            
            # Get prediction-based signal
            pred_signal = apply_policy(
                pair=pair,
                inference_entry=results[pair],
                portfolio_value=total_value,
                position_limit_fraction=args.position_limit,
                current_position_notional=current_position,
                config=policy_config,
            )
            
            # Hybrid mode: combine with MA signal
            if use_hybrid:
                ma_signal = _get_ma_signal(
                    pair=pair,
                    portfolio_value=total_value,
                    position_limit_fraction=args.position_limit,
                    current_position_notional=current_position,
                    ma_fast_period=ma_fast_period,
                    ma_slow_period=ma_slow_period,
                    ma_state=ma_state,
                    db_path=db_path,
                    ma_weight=hybrid_ma_weight,
                )
                
                # Combine signals with weights
                signal = _combine_hybrid_signals(
                    pred_signal=pred_signal,
                    ma_signal=ma_signal,
                    pred_weight=hybrid_prediction_weight,
                    ma_weight=hybrid_ma_weight,
                    portfolio_value=total_value,
                    position_limit_fraction=args.position_limit,
                    current_position_notional=current_position,
                )
                logger.info(f"Hybrid policy: Prediction={pred_signal['action']} (${pred_signal['notional']:.2f}), MA={ma_signal['action']} (${ma_signal['notional']:.2f}), Combined={signal['action']} (${signal['notional']:.2f})")
            else:
                signal = pred_signal
                logger.info(f"Policy decision: {signal}")

            if not args.paper:
                trade = execute_trade(
                    pair=pair,
                    action=signal["action"],
                    notional=signal["notional"],
                    config=execution_config,
                )
                if trade.get("status") != "hold" and trade.get("price"):
                    position_manager.apply_trade(
                        pair=pair,
                        side=trade.get("side", "BUY"),
                        notional=trade.get("notional", 0.0),
                        price=trade.get("price", 0.0),
                    )
            else:
                if signal["action"] != "hold" and signal["notional"] > 0:
                    try:
                        price = position_manager.prices.get(pair) or get_last_price(pair)
                    except Exception:
                        price = None
                    if price:
                        position_manager.apply_simulated(
                            pair=pair,
                            action=signal["action"],
                            notional=signal["notional"],
                            price=price,
                        )

        logger.info(f"Sleeping for {args.loop_interval} seconds.")
        time.sleep(args.loop_interval)


def _get_ma_signal(
    pair: str,
    portfolio_value: float,
    position_limit_fraction: float,
    current_position_notional: float,
    ma_fast_period: int,
    ma_slow_period: int,
    ma_state: dict,
    db_path: Path,
    ma_weight: float,
) -> Dict[str, Any]:
    """Get MA crossover signal for a pair."""
    try:
        # Get price history from database
        conn = sqlite3.connect(db_path)
        query = """
            SELECT close, as_of 
            FROM horus_prices_1h 
            WHERE pair = ? 
            ORDER BY as_of DESC 
            LIMIT ?
        """
        min_periods = max(ma_slow_period, 26)
        df = pd.read_sql(query, conn, params=(pair, min_periods * 2))
        conn.close()
        
        if len(df) < min_periods:
            return {"pair": pair, "action": "hold", "notional": 0.0, "reason": "insufficient_data"}
        
        # Calculate MAs
        prices = df['close'].astype(float).iloc[::-1]  # Reverse to chronological order
        ma_fast, ma_slow = calculate_moving_averages(prices, fast=ma_fast_period, slow=ma_slow_period)
        
        if len(ma_fast) == 0 or len(ma_slow) == 0:
            return {"pair": pair, "action": "hold", "notional": 0.0, "reason": "ma_calc_failed"}
        
        current_price = float(prices.iloc[-1])
        current_ma_fast = float(ma_fast.iloc[-1])
        current_ma_slow = float(ma_slow.iloc[-1])
        
        # Check for crossover
        last_ma_fast = ma_state[pair]['last_ma_fast']
        last_ma_slow = ma_state[pair]['last_ma_slow']
        
        ma_signal = {"pair": pair, "action": "hold", "notional": 0.0, "reason": "no_crossover"}
        
        if last_ma_fast is not None and last_ma_slow is not None:
            # Golden cross: fast MA crosses above slow MA
            if last_ma_fast <= last_ma_slow and current_ma_fast > current_ma_slow:
                if current_price > current_ma_fast:
                    # Calculate notional (50% of base risk)
                    base_notional = portfolio_value * 0.05 * hybrid_ma_weight  # 5% base risk * MA weight
                    max_position = portfolio_value * position_limit_fraction
                    remaining_limit = max(0.0, max_position - current_position_notional)
                    notional = min(base_notional, remaining_limit)
                    if notional > 0:
                        ma_signal = {"pair": pair, "action": "buy", "notional": notional, "reason": "ma_golden_cross"}
            
            # Death cross: fast MA crosses below slow MA
            elif last_ma_fast >= last_ma_slow and current_ma_fast < current_ma_slow:
                if current_price < current_ma_fast:
                    # Sell current position (up to MA weight portion)
                    base_notional = portfolio_value * 0.05 * ma_weight
                    notional = min(base_notional, current_position_notional)
                    if notional > 0:
                        ma_signal = {"pair": pair, "action": "sell", "notional": notional, "reason": "ma_death_cross"}
        
        # Update state
        ma_state[pair]['last_ma_fast'] = current_ma_fast
        ma_state[pair]['last_ma_slow'] = current_ma_slow
        
        return ma_signal
    except Exception as e:
        logger.warning(f"[{pair}] Error getting MA signal: {e}")
        return {"pair": pair, "action": "hold", "notional": 0.0, "reason": f"error: {str(e)}"}


def _combine_hybrid_signals(
    pred_signal: Dict[str, Any],
    ma_signal: Dict[str, Any],
    pred_weight: float,
    ma_weight: float,
    portfolio_value: float,
    position_limit_fraction: float,
    current_position_notional: float,
) -> Dict[str, Any]:
    """Combine prediction and MA signals with weights."""
    # Scale prediction notional by weight
    pred_notional = pred_signal.get("notional", 0.0) * pred_weight
    ma_notional = ma_signal.get("notional", 0.0) * ma_weight
    
    # Determine combined action
    pred_action = pred_signal.get("action", "hold")
    ma_action = ma_signal.get("action", "hold")
    
    # If both agree on direction, combine
    if pred_action == ma_action and pred_action != "hold":
        combined_notional = pred_notional + ma_notional
        max_position = portfolio_value * position_limit_fraction
        if pred_action == "buy":
            remaining_limit = max(0.0, max_position - current_position_notional)
            combined_notional = min(combined_notional, remaining_limit)
        else:  # sell
            combined_notional = min(combined_notional, current_position_notional)
        
        return {
            "pair": pred_signal["pair"],
            "action": pred_action,
            "notional": max(0.0, combined_notional),
            "reason": f"hybrid_{pred_action}_both_agree",
            "pred_signal": pred_signal,
            "ma_signal": ma_signal,
        }
    # If only one signals, use that one
    elif pred_action != "hold":
        return {
            "pair": pred_signal["pair"],
            "action": pred_action,
            "notional": pred_notional,
            "reason": f"hybrid_{pred_action}_prediction_only",
            "pred_signal": pred_signal,
            "ma_signal": ma_signal,
        }
    elif ma_action != "hold":
        return {
            "pair": ma_signal["pair"],
            "action": ma_action,
            "notional": ma_notional,
            "reason": f"hybrid_{ma_action}_ma_only",
            "pred_signal": pred_signal,
            "ma_signal": ma_signal,
        }
    # Both hold
    else:
        return {
            "pair": pred_signal["pair"],
            "action": "hold",
            "notional": 0.0,
            "reason": "hybrid_both_hold",
            "pred_signal": pred_signal,
            "ma_signal": ma_signal,
        }


if __name__ == '__main__':
    main()

