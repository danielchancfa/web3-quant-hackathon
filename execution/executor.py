"""
Execution module that wraps Roostoo API calls with risk checks and logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any

from config import get_config
from data_pipeline.roostoo_api import place_order, get_balance, get_ticker

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    slippage_buffer: float = 0.001  # 0.1% price buffer for limit orders
    max_retries: int = 3
    retry_wait_seconds: float = 1.0


def get_last_price(pair: str) -> float:
    ticker = get_ticker(pair)
    data = {}
    if isinstance(ticker, dict):
        data = ticker.get("Data", {}).get(pair, {})
        if not data and "result" in ticker:
            data = ticker["result"].get(pair, {})
    for key in ("last", "LastPrice", "price", "close"):
        price = data.get(key)
        if price not in (None, "", 0):
            try:
                return float(price)
            except (TypeError, ValueError):
                continue
    raise RuntimeError(f"Could not fetch last price for {pair}: raw={ticker}")


def place_market_order(pair: str, side: str, quantity: float) -> Dict[str, Any]:
    return place_order(pair, side.upper(), quantity, price=None, order_type="MARKET")


def execute_trade(
    pair: str,
    action: str,
    notional: float,
    config: ExecutionConfig | None = None,
) -> Dict[str, Any]:
    """
    Execute a trade if notional > 0. Returns summary with price/quantity.
    """
    config = config or ExecutionConfig()
    if action == "hold" or notional <= 0:
        return {"status": "hold", "pair": pair, "notional": 0.0}

    price = get_last_price(pair)
    quantity = notional / price
    side = "BUY" if action == "buy" else "SELL"

    logger.info(f"[{pair}] Executing {side} qty={quantity:.6f} (~${notional:.2f}) at price {price:.2f}")
    result = place_market_order(pair, side, quantity)
    logger.info(f"[{pair}] Order result: {result}")

    return {
        "status": "submitted" if result else "error",
        "pair": pair,
        "side": side,
        "notional": notional,
        "price": price,
        "quantity": quantity,
        "api_response": result,
    }

