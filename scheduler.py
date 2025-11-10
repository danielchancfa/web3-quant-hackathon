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
            signal = apply_policy(
                pair=pair,
                inference_entry=results[pair],
                portfolio_value=total_value,
                position_limit_fraction=args.position_limit,
                current_position_notional=current_position,
                config=policy_config,
            )
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


if __name__ == '__main__':
    main()

