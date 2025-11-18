"""
Scheduler that ties inference, policy, and execution together.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

from config import get_config
from inference_service import InferenceService
from execution.policy import apply_policy, PolicyConfig
from execution.executor import execute_trade, ExecutionConfig, get_last_price
from execution.position_manager import PositionManager
from tools.technical_indicators import calculate_moving_averages
from data_pipeline.historical_data_pipeline import main as update_data_pipeline
import sqlite3
import pandas as pd

# Configure logging to both file and console
def setup_logging(log_file: str = "scheduler.log"):
    """Setup logging to both file and console."""
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Clear any existing handlers
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

# Setup logging
logger = setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trading loop.")
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--pairs', type=str, required=False, default=None,
                        help="Comma-separated pairs, e.g., BTC/USD,ETH/USD. Optional in hybrid mode (auto-detects all pairs)")
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
    parser.add_argument('--ma_period', type=int, default=16, help="MA period for trend filter (default: 16 hours)")
    parser.add_argument('--use_atr_stops', action='store_true',
                        help="Use ATR-based adaptive stop-loss/take-profit (default: True, enabled by default)")
    parser.add_argument('--no_atr_stops', action='store_true',
                        help="Disable ATR-based stops, use fixed percentage instead")
    parser.add_argument('--atr_period', type=int, default=14, help="ATR period (default: 14)")
    parser.add_argument('--atr_stop_multiplier', type=float, default=1.5,
                        help="ATR multiplier for stop loss (default: 1.5)")
    parser.add_argument('--atr_target_multiplier', type=float, default=2.5,
                        help="ATR multiplier for take profit (default: 2.5)")
    parser.add_argument('--min_stop_pct', type=float, default=0.005,
                        help="Minimum stop loss % even with ATR (default: 0.5%)")
    parser.add_argument('--max_stop_pct', type=float, default=0.05,
                        help="Maximum stop loss % even with ATR (default: 5%)")
    return parser.parse_args()


def _get_all_checkpoint_pairs(checkpoint_dir: Path) -> list:
    """Get all pairs that have checkpoints."""
    pairs = []
    if not checkpoint_dir.exists():
        return pairs
    
    for item in checkpoint_dir.iterdir():
        if item.is_dir():
            # Skip non-pair directories
            if item.name in ['final', 'next', 'run', 'binary']:
                continue
            # Convert directory name to pair format (e.g., ADA_USD -> ADA/USD)
            pair_name = item.name.replace('_', '/')
            pairs.append(pair_name)
    
    return sorted(pairs)


def _get_all_database_pairs(db_path: Path) -> list:
    """Get all pairs that have price data in database."""
    pairs = []
    if not db_path.exists():
        return pairs
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT pair FROM ohlcv WHERE interval = '1h' ORDER BY pair")
        pairs = [row[0] for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        logger.warning(f"Error querying database for pairs: {e}")
    
    return pairs


def main() -> None:
    args = parse_args()
    config = get_config()
    db_path = Path(config.db_path)
    checkpoint_dir = Path(args.checkpoint_dir)
    
    # Auto-detect all available pairs
    all_checkpoint_pairs = _get_all_checkpoint_pairs(checkpoint_dir)
    all_database_pairs = _get_all_database_pairs(db_path)
    
    # For hybrid strategy: use intersection of checkpoints and database pairs
    # This ensures both models can trade all pairs
    use_hybrid = args.hybrid if hasattr(args, 'hybrid') else False
    
    if use_hybrid:
        # Use intersection: pairs that both models can trade
        aligned_pairs = sorted(list(set(all_checkpoint_pairs) & set(all_database_pairs)))
        
        # In hybrid mode, --pairs is optional and ignored (we auto-detect)
        # But if provided, we can use it as a filter to limit pairs
        if args.pairs:
            requested_pairs = [p.strip() for p in args.pairs.split(',') if p.strip()]
            # Filter aligned pairs to only include requested ones
            final_pairs = sorted([p for p in aligned_pairs if p in requested_pairs])
            logger.info(f"🔍 Auto-detected {len(all_checkpoint_pairs)} pairs with checkpoints")
            logger.info(f"🔍 Auto-detected {len(all_database_pairs)} pairs with price data")
            logger.info(f"🔍 Aligned pairs (both models can trade): {len(aligned_pairs)}")
            logger.info(f"🔍 Filtered to requested pairs: {len(final_pairs)}")
            logger.info(f"✅ Using {len(final_pairs)} pairs for hybrid strategy: {final_pairs}")
        else:
            # No --pairs specified, use all aligned pairs
            final_pairs = aligned_pairs
            logger.info(f"🔍 Auto-detected {len(all_checkpoint_pairs)} pairs with checkpoints")
            logger.info(f"🔍 Auto-detected {len(all_database_pairs)} pairs with price data")
            logger.info(f"✅ Using all {len(final_pairs)} aligned pairs for hybrid strategy: {final_pairs}")
        
        pairs = final_pairs
    else:
        # For non-hybrid: require --pairs argument
        if not args.pairs:
            logger.error("--pairs argument is required when not using --hybrid mode")
            raise ValueError("--pairs is required in non-hybrid mode")
        
        requested_pairs = [p.strip() for p in args.pairs.split(',') if p.strip()]
        pairs = requested_pairs
        logger.info(f"Using requested pairs: {pairs}")

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
    
    # Include any pairs with open positions, even if not in the main pairs list
    # This ensures we can close old positions (e.g., ZEC/USD from before)
    use_hybrid = args.hybrid if hasattr(args, 'hybrid') else False
    
    # Initialize position manager with all pairs we'll trade
    position_manager = PositionManager(pairs)
    
    # Hybrid mode: MA strategy settings
    hybrid_prediction_weight = args.hybrid_prediction_weight if hasattr(args, 'hybrid_prediction_weight') else 0.5
    hybrid_ma_weight = args.hybrid_ma_weight if hasattr(args, 'hybrid_ma_weight') else 0.5
    ma_period = args.ma_period if hasattr(args, 'ma_period') else 16
    
    # Also initialize MA state for any pairs that might have open positions
    # (will be updated dynamically as we discover them)

    while True:
        # Step 1: Update data pipeline first (fetch fresh data)
        logger.info("=" * 80)
        logger.info("STEP 1: Updating data pipeline...")
        logger.info("=" * 80)
        try:
            update_data_pipeline()
            logger.info("✅ Data pipeline update complete")
        except Exception as e:
            logger.error(f"❌ Error updating data pipeline: {e}")
            logger.warning("Continuing with existing data in database...")
        
        # Step 2: Refresh positions
        logger.info("=" * 80)
        logger.info("STEP 2: Refreshing positions...")
        logger.info("=" * 80)
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

        # Step 3: Get predictions
        logger.info("=" * 80)
        logger.info("STEP 3: Getting predictions...")
        logger.info("=" * 80)
        results = service.run_once()
        logger.info(json.dumps(results, indent=2))
        logger.info(f"Portfolio value: {total_value:.2f}")

        # Get all pairs we need to process:
        # 1. Pairs from results (prediction model can trade)
        # 2. Pairs with open positions (need to monitor for stop/take-profit)
        prediction_pairs = list(results.keys())
        open_position_pairs = [p for p in position_manager.pair_notionals.keys() if position_manager.get_pair_notional(p) > 0]
        
        # Combine: all pairs we should process
        all_pairs_to_process = sorted(list(set(prediction_pairs + open_position_pairs)))
        
        logger.info(f"🔍 Processing {len(all_pairs_to_process)} pairs:")
        logger.info(f"   • Prediction model pairs: {len(prediction_pairs)}")
        logger.info(f"   • Pairs with open positions: {len(open_position_pairs)}")
        logger.info(f"   • Total pairs to process: {all_pairs_to_process}")
        
        # Process each pair
        for pair in all_pairs_to_process:
            current_position = position_manager.get_pair_notional(pair)
            
            # Check if we have prediction results for this pair
            has_prediction = pair in results
            
            # Check stop-loss/take-profit for existing positions
            if current_position > 0:
                # ATR is enabled by default unless --no_atr_stops is set
                use_atr = not getattr(args, 'no_atr_stops', False)
                if hasattr(args, 'use_atr_stops') and args.use_atr_stops:
                    use_atr = True
                
                stop_check = position_manager.check_stop_take_profit(
                    pair=pair,
                    use_atr=use_atr,
                    atr_period=getattr(args, 'atr_period', 14),
                    atr_stop_multiplier=getattr(args, 'atr_stop_multiplier', 1.5),
                    atr_target_multiplier=getattr(args, 'atr_target_multiplier', 2.5),
                    db_path=db_path,
                    min_stop_pct=getattr(args, 'min_stop_pct', 0.005),
                    max_stop_pct=getattr(args, 'max_stop_pct', 0.05),
                )
                if stop_check["should_close"]:
                    logger.info(
                        f"[{pair}] Stop/Take-profit triggered: {stop_check['reason']} "
                        f"(Entry: ${stop_check.get('entry_price', 0):.4f}, "
                        f"Current: ${stop_check.get('current_price', 0):.4f}, "
                        f"Change: {stop_check.get('price_change_pct', 0)*100:.2f}%)"
                    )
                    # Close position
                    close_notional = stop_check["notional"]
                    if not args.paper:
                        trade = execute_trade(
                            pair=pair,
                            action="sell",
                            notional=close_notional,
                            config=execution_config,
                        )
                        if trade.get("status") != "hold" and trade.get("price"):
                            position_manager.apply_trade(
                                pair=pair,
                                side="SELL",
                                notional=trade.get("notional", close_notional),
                                price=trade.get("price", 0.0),
                            )
                            logger.info(f"[{pair}] Position closed due to stop/take-profit")
                    else:
                        price = position_manager.prices.get(pair) or get_last_price(pair)
                        position_manager.apply_simulated(pair, "sell", close_notional, price)
                        logger.info(f"[{pair}] Paper: Position closed due to stop/take-profit")
                    continue  # Skip to next pair after closing
            
            # Get prediction-based signal (only if we have prediction results)
            if has_prediction:
                pred_signal = apply_policy(
                    pair=pair,
                    inference_entry=results[pair],
                    portfolio_value=total_value,
                    position_limit_fraction=args.position_limit,
                    current_position_notional=current_position,
                    config=policy_config,
                )
            else:
                # No prediction results (e.g., old position without checkpoint)
                # Only monitor for stop/take-profit, don't generate new signals
                logger.info(f"[{pair}] No prediction model available - only monitoring existing position")
                pred_signal = {"pair": pair, "action": "hold", "notional": 0.0, "reason": "no_prediction_model"}
            
            # Hybrid mode: combine with MA signal
            if use_hybrid:
                # Get MA signal (always try, even if no prediction)
                ma_signal = _get_ma_signal(
                    pair=pair,
                    ma_period=ma_period,
                    db_path=db_path,
                )
                
                # Combine signals with weights
                if has_prediction:
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
                    # No prediction model: MA-only logic
                    # SELL if MA downtrend (price < 16h MA) and we have position
                    if ma_signal['action'] == 'sell' and current_position > 0:
                        signal = {
                            "pair": pair,
                            "action": "sell",
                            "notional": current_position,
                            "reason": "ma_only_downtrend",
                        }
                    else:
                        # MA uptrend but no prediction model → can't buy without prediction
                        signal = {
                            "pair": pair,
                            "action": "hold",
                            "notional": 0.0,
                            "reason": "ma_only_no_prediction",
                        }
                    logger.info(f"MA-only policy (no prediction model): MA={ma_signal['action']}, Combined={signal['action']} (${signal['notional']:.2f})")
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
    ma_period: int,
    db_path: Path,
) -> Dict[str, Any]:
    """
    Get MA trend signal for a pair using single 16-hour MA.
    
    Returns:
    - "buy" if price > MA (uptrend)
    - "sell" if price < MA (downtrend)
    - "hold" if insufficient data or error
    """
    try:
        # Get price history from database
        conn = sqlite3.connect(db_path)
        query = """
            SELECT close, timestamp as as_of 
            FROM ohlcv 
            WHERE pair = ? AND interval = '1h'
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        min_periods = max(ma_period, 20)
        df = pd.read_sql(query, conn, params=(pair, min_periods * 2))
        conn.close()
        
        if len(df) < min_periods:
            return {"pair": pair, "action": "hold", "notional": 0.0, "reason": "insufficient_data"}
        
        # Calculate single MA
        prices = df['close'].astype(float).iloc[::-1]  # Reverse to chronological order
        ma_values = prices.rolling(window=ma_period).mean()
        
        if len(ma_values) == 0 or pd.isna(ma_values.iloc[-1]):
            return {"pair": pair, "action": "hold", "notional": 0.0, "reason": "ma_calc_failed"}
        
        current_price = float(prices.iloc[-1])
        current_ma = float(ma_values.iloc[-1])
        
        # Simple trend filter: price above MA = uptrend, below MA = downtrend
        if current_price > current_ma:
            return {"pair": pair, "action": "buy", "notional": 0.0, "reason": "ma_uptrend"}
        else:
            return {"pair": pair, "action": "sell", "notional": 0.0, "reason": "ma_downtrend"}
        
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
    """
    Combine prediction and MA signals in hybrid mode.

    NEW LOGIC:
    
    BUY:
    - Price > 16h MA (uptrend) AND prediction = BUY → BUY
    - Uses prediction model's notional (scaled by pred_weight)
    
    SELL:
    - Price < 16h MA (downtrend) OR prediction = SELL → SELL 100% of position
    - Always closes full position when either condition is met
    - Never shorts (requires existing position)
    """
    # Determine actions
    pred_action = pred_signal.get("action", "hold")
    ma_action = ma_signal.get("action", "hold")
    
    # MA trend: "buy" means price > MA (uptrend), "sell" means price < MA (downtrend)
    ma_uptrend = (ma_action == "buy")
    ma_downtrend = (ma_action == "sell")
    pred_wants_buy = (pred_action == "buy")
    pred_wants_sell = (pred_action == "sell")

    # --- SELL LOGIC: Either MA downtrend OR prediction SELL → close full position ---
    if ma_downtrend or pred_wants_sell:
        # Never go short: require existing position
        if current_position_notional <= 0:
            return {
                "pair": pred_signal.get("pair") or ma_signal.get("pair"),
                "action": "hold",
                "notional": 0.0,
                "reason": "hybrid_sell_no_position",
                "pred_signal": pred_signal,
                "ma_signal": ma_signal,
            }

        # Either condition met → close full position
        sell_notional = current_position_notional
        if ma_downtrend and pred_wants_sell:
            reason = "hybrid_sell_both_ma_downtrend_and_prediction"
        elif ma_downtrend:
            reason = "hybrid_sell_ma_downtrend"
        else:
            reason = "hybrid_sell_prediction"

        return {
            "pair": pred_signal.get("pair") or ma_signal.get("pair"),
            "action": "sell",
            "notional": max(0.0, sell_notional),
            "reason": reason,
            "pred_signal": pred_signal,
            "ma_signal": ma_signal,
        }

    # --- BUY LOGIC: MA uptrend AND prediction BUY → BUY ---
    if ma_uptrend and pred_wants_buy:
        # Both conditions met → use prediction model's notional (scaled by weight)
        pred_notional = pred_signal.get("notional", 0.0) * pred_weight
        max_position = portfolio_value * position_limit_fraction
        remaining_limit = max(0.0, max_position - current_position_notional)
        buy_notional = min(pred_notional, remaining_limit)
        
        if buy_notional > 0:
            return {
                "pair": pred_signal["pair"],
                "action": "buy",
                "notional": buy_notional,
                "reason": "hybrid_buy_ma_uptrend_and_prediction",
                "pred_signal": pred_signal,
                "ma_signal": ma_signal,
            }
    
    # --- HOLD: Conditions not met ---
    return {
        "pair": pred_signal.get("pair") or ma_signal.get("pair"),
        "action": "hold",
        "notional": 0.0,
        "reason": "hybrid_hold_conditions_not_met",
        "pred_signal": pred_signal,
        "ma_signal": ma_signal,
    }


if __name__ == '__main__':
    main()

