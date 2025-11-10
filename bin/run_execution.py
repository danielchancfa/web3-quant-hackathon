"""
Read inference results, recompute actions using current positions, and execute trades.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config
from execution.policy import apply_policy, PolicyConfig
from execution.executor import execute_trade, ExecutionConfig, get_last_price
from execution.position_manager import PositionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute trades based on stored inference.")
    parser.add_argument("--input", type=str, default="artifacts/inference_output.json",
                        help="Path to inference JSON output.")
    parser.add_argument("--position_limit", type=float, default=0.3,
                        help="Max fraction of portfolio in any single asset.")
    parser.add_argument("--paper", action="store_true", help="Simulate trades without sending orders.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference_file = Path(args.input)
    if not inference_file.exists():
        raise FileNotFoundError(f"Inference file {inference_file} not found.")

    payload = json.loads(inference_file.read_text())
    pairs = payload["pairs"]
    inference_results = payload["results"]

    logger.info("Loaded inference results for pairs: %s", ", ".join(pairs))

    position_manager = PositionManager(pairs)
    position_manager.refresh()
    portfolio_value = position_manager.total_value()
    logger.info(f"Portfolio value: {portfolio_value:.2f}")

    policy_config = PolicyConfig()
    execution_config = ExecutionConfig()

    for pair in pairs:
        entry = inference_results[pair]
        current_position = position_manager.get_pair_notional(pair)
        signal = apply_policy(
            pair=pair,
            inference_entry=entry,
            portfolio_value=portfolio_value,
            position_limit_fraction=args.position_limit,
            current_position_notional=current_position,
            config=policy_config,
        )
        logger.info(f"[{pair}] Policy decision: {signal}")

        if signal["action"] == "hold" or signal["notional"] <= 0:
            continue

        if args.paper:
            price = position_manager.prices.get(pair)
            if not price:
                try:
                    price = get_last_price(pair)
                except Exception:
                    price = None
            if price:
                position_manager.apply_simulated(
                    pair=pair,
                    action=signal["action"],
                    notional=signal["notional"],
                    price=price,
                )
            logger.info(f"[{pair}] Paper trade simulated.")
            continue

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


if __name__ == "__main__":
    main()

