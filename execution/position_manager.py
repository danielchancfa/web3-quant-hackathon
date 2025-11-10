"""
Position manager that tracks cash and per-pair holdings based on Roostoo balances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any

from data_pipeline.roostoo_api import get_balance
from execution.executor import get_last_price

logger = logging.getLogger(__name__)


def _parse_available(entry: dict) -> float:
    for key in ("availableBalance", "available", "balance", "Amount", "amount", "free", "Free"):
        if key in entry and entry[key] is not None:
            try:
                return float(entry[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def _split_pair(pair: str) -> Tuple[str, str]:
    if "/" in pair:
        base, quote = pair.split("/")
        return base.upper(), quote.upper()
    return pair.upper(), "USD"


@dataclass
class PositionManager:
    pairs: List[str]
    cash_usd: float = 0.0
    asset_qty: Dict[str, float] = field(default_factory=dict)
    prices: Dict[str, float] = field(default_factory=dict)
    pair_notionals: Dict[str, float] = field(default_factory=dict)

    def refresh(self) -> None:
        """Fetch balances from Roostoo and compute notionals for tracked pairs."""
        resp = get_balance()
        if not isinstance(resp, dict):
            logger.warning("get_balance returned non-dict; keeping previous positions.")
            return

        if logger.isEnabledFor(logging.DEBUG):
            import json
            logger.debug("Raw balance response:\n%s", json.dumps(resp, indent=2))

        data = resp.get("Data") or resp.get("data") or resp
        cash, assets = self._extract_assets(data)
        if cash == 0.0 and not assets:
            logger.warning("Parsed zero balances; raw response keys: %s", list(resp.keys()))

        self.cash_usd = cash
        self.asset_qty = assets
        self.prices = {}
        self.pair_notionals = {}

        for pair in self.pairs:
            base, _ = _split_pair(pair)
            qty = assets.get(base, 0.0)
            if qty <= 0:
                self.pair_notionals[pair] = 0.0
                continue
            try:
                price = get_last_price(pair)
                self.prices[pair] = price
                self.pair_notionals[pair] = qty * price
            except Exception as exc:
                logger.warning(f"Could not fetch price for {pair}: {exc}")
                self.pair_notionals[pair] = 0.0

    def total_value(self) -> float:
        total = self.cash_usd
        total += sum(self.pair_notionals.values())
        return total

    def get_pair_notional(self, pair: str) -> float:
        return self.pair_notionals.get(pair, 0.0)

    def apply_trade(self, pair: str, side: str, notional: float, price: float) -> None:
        """Adjust holdings after a real trade."""
        base, _ = _split_pair(pair)
        qty_change = notional / price if price > 0 else 0.0
        if side.upper() == "BUY":
            self.cash_usd -= notional
            self.asset_qty[base] = self.asset_qty.get(base, 0.0) + qty_change
        else:
            self.cash_usd += notional
            self.asset_qty[base] = self.asset_qty.get(base, 0.0) - qty_change
            if self.asset_qty[base] < 0:
                self.asset_qty[base] = 0.0
        # Update notional cache
        self.prices[pair] = price
        self.pair_notionals[pair] = self.asset_qty.get(base, 0.0) * price

    def apply_simulated(self, pair: str, action: str, notional: float, price: float) -> None:
        """Simulate position updates in paper trading mode."""
        if action == "hold" or notional <= 0:
            return
        side = "BUY" if action == "buy" else "SELL"
        self.apply_trade(pair, side, notional, price)

    def _extract_assets(self, data: Any, parent_symbol: str | None = None) -> Tuple[float, Dict[str, float]]:
        cash_total = 0.0
        assets_total: Dict[str, float] = {}

        if data is None:
            return cash_total, assets_total
        if isinstance(data, dict):
            qty = _parse_available(data)
            symbol = data.get("symbol") or data.get("asset") or data.get("token") or parent_symbol
            if qty > 0 and symbol:
                symbol_u = str(symbol).upper()
                if symbol_u in ("USD", "USDT", "CASH"):
                    cash_total += qty
                else:
                    assets_total[symbol_u] = assets_total.get(symbol_u, 0.0) + qty
            else:
                for key, value in data.items():
                    sub_cash, sub_assets = self._extract_assets(value, parent_symbol=key)
                    cash_total += sub_cash
                    for sym, qty in sub_assets.items():
                        assets_total[sym] = assets_total.get(sym, 0.0) + qty
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    symbol = item.get("symbol") or item.get("asset") or parent_symbol
                    sub_cash, sub_assets = self._extract_assets(item, parent_symbol=symbol)
                    cash_total += sub_cash
                    for sym, qty in sub_assets.items():
                        assets_total[sym] = assets_total.get(sym, 0.0) + qty

        return cash_total, assets_total



