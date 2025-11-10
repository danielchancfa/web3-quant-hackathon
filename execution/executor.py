"""
Execution module that wraps Roostoo API calls with risk checks and logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from functools import lru_cache
from typing import Dict, Any

from config import get_config
from data_pipeline.roostoo_api import (
    place_order,
    get_balance,
    get_ticker,
    get_exchange_info,
)

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


@lru_cache(maxsize=64)
def _get_pair_constraints(pair: str) -> Dict[str, Decimal]:
    """
    Fetch and cache precision / minimum order constraints for a pair.
    """
    info = get_exchange_info() or {}
    trade_pairs = info.get("TradePairs") or {}
    meta = trade_pairs.get(pair)
    if not meta:
        raise RuntimeError(f"Pair {pair} not found in exchange info: {info.keys()}")

    amount_precision = int(meta.get("AmountPrecision", 0))
    price_precision = int(meta.get("PricePrecision", 0))
    min_order = Decimal(str(meta.get("MiniOrder", "0") or "0"))
    if min_order <= 0:
        min_order = Decimal("1")  # Exchange enforces $1 USD minimum notional

    qty_step = Decimal("1") / (Decimal("10") ** amount_precision) if amount_precision >= 0 else Decimal("0.00000001")
    price_step = Decimal("1") / (Decimal("10") ** price_precision) if price_precision >= 0 else Decimal("0.00000001")

    return {
        "qty_step": qty_step,
        "price_step": price_step,
        "min_notional": min_order,
    }


def _quantize_order(pair: str, notional: Decimal, raw_price: Decimal) -> Dict[str, Decimal]:
    """
    Adjust quantity and notional to satisfy exchange precision and minimums.
    """
    constraints = _get_pair_constraints(pair)
    qty_step = constraints["qty_step"]
    price_step = constraints["price_step"]
    min_notional = constraints["min_notional"]

    price = raw_price.quantize(price_step, rounding=ROUND_DOWN)
    if price <= 0:
        raise RuntimeError(f"Invalid price for {pair}: {raw_price}")

    qty = (notional / price).quantize(qty_step, rounding=ROUND_DOWN)
    if qty <= 0:
        qty = (notional / price).quantize(qty_step, rounding=ROUND_UP)
    if qty <= 0:
        qty = qty_step

    adjusted_notional = (qty * price).quantize(price_step, rounding=ROUND_DOWN)

    if min_notional > 0 and adjusted_notional < min_notional:
        qty = (min_notional / price).quantize(qty_step, rounding=ROUND_UP)
        adjusted_notional = (qty * price).quantize(price_step, rounding=ROUND_UP)

    if qty <= 0:
        raise RuntimeError(
            f"Unable to quantize {pair} order with notional {notional} and price {price}; "
            f"min_notional={min_notional}, qty_step={qty_step}"
        )

    return {"price": price, "quantity": qty, "notional": adjusted_notional}


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

    raw_price = Decimal(str(get_last_price(pair)))
    target_notional = Decimal(str(notional))
    quantized = _quantize_order(pair, target_notional, raw_price)

    quantity = float(quantized["quantity"])
    price = float(quantized["price"])
    actual_notional = float(quantized["notional"])
    side = "BUY" if action == "buy" else "SELL"

    logger.info(
        f"[{pair}] Executing {side} qty={quantity:.6f} (~${actual_notional:.2f}) at price {price:.6f} "
        f"(requested notional ${notional:.2f})"
    )
    result = place_market_order(pair, side, quantity)
    logger.info(f"[{pair}] Order result: {result}")

    return {
        "status": "submitted" if result else "error",
        "pair": pair,
        "side": side,
        "notional": actual_notional,
        "price": price,
        "quantity": quantity,
        "api_response": result,
    }

