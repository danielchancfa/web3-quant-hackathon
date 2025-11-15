"""
Position manager that tracks cash and per-pair holdings based on Roostoo balances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any

from data_pipeline.roostoo_api import get_balance
from execution.executor import get_last_price
from tools.technical_indicators import calculate_atr
import sqlite3
import pandas as pd

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
    # Track entry prices for stop-loss/take-profit monitoring
    entry_prices: Dict[str, float] = field(default_factory=dict)  # pair -> entry_price

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
            current_qty = self.asset_qty.get(base, 0.0)
            new_qty = current_qty + qty_change
            self.asset_qty[base] = new_qty
            # Update entry price (weighted average)
            if current_qty > 0:
                # Weighted average of existing and new entry
                old_entry = self.entry_prices.get(pair, price)
                total_notional_old = current_qty * old_entry
                total_notional_new = qty_change * price
                self.entry_prices[pair] = (total_notional_old + total_notional_new) / new_qty if new_qty > 0 else price
            else:
                # New position
                self.entry_prices[pair] = price
        else:
            self.cash_usd += notional
            self.asset_qty[base] = self.asset_qty.get(base, 0.0) - qty_change
            if self.asset_qty[base] <= 0:
                self.asset_qty[base] = 0.0
                # Clear entry price when position closed
                if pair in self.entry_prices:
                    del self.entry_prices[pair]
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
    
    def check_stop_take_profit(
        self, 
        pair: str, 
        stop_pct: float = None, 
        target_pct: float = None,
        use_atr: bool = True,
        atr_period: int = 14,
        atr_stop_multiplier: float = 1.5,
        atr_target_multiplier: float = 2.5,
        db_path: Path = None,
        min_stop_pct: float = 0.005,  # Minimum 0.5% stop even with ATR
        max_stop_pct: float = 0.05,   # Maximum 5% stop even with ATR
    ) -> Dict[str, Any]:
        """
        Check if position should be closed due to stop-loss or take-profit.
        
        Uses ATR-based adaptive stops by default, or fixed percentage if use_atr=False.
        
        Returns:
            dict with 'should_close' (bool), 'reason' (str), 'notional' (float)
        """
        if pair not in self.entry_prices:
            return {"should_close": False, "reason": "no_entry_price", "notional": 0.0}
        
        current_notional = self.get_pair_notional(pair)
        if current_notional <= 0:
            return {"should_close": False, "reason": "no_position", "notional": 0.0}
        
        entry_price = self.entry_prices[pair]
        current_price = self.prices.get(pair)
        
        if current_price is None or current_price <= 0:
            return {"should_close": False, "reason": "no_current_price", "notional": 0.0}
        
        # Calculate stop and target prices
        if use_atr and db_path:
            try:
                # Get price history for ATR calculation
                conn = sqlite3.connect(db_path)
                query = """
                    SELECT high, low, close, timestamp as as_of 
                    FROM ohlcv 
                    WHERE pair = ? AND interval = '1h'
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """
                df = pd.read_sql(query, conn, params=(pair, atr_period * 2))
                conn.close()
                
                if len(df) >= atr_period:
                    # Reverse to chronological order
                    df = df.iloc[::-1]
                    high = df['high'].astype(float)
                    low = df['low'].astype(float)
                    close = df['close'].astype(float)
                    
                    # Calculate ATR
                    atr = calculate_atr(high, low, close, period=atr_period)
                    current_atr = float(atr.iloc[-1]) if len(atr) > 0 and not pd.isna(atr.iloc[-1]) else None
                    
                    if current_atr and current_atr > 0:
                        # ATR-based stops
                        stop_distance = current_atr * atr_stop_multiplier
                        target_distance = current_atr * atr_target_multiplier
                        
                        # Convert to percentages and clamp
                        stop_pct_atr = stop_distance / entry_price
                        target_pct_atr = target_distance / entry_price
                        
                        # Clamp to min/max
                        stop_pct_atr = max(min_stop_pct, min(stop_pct_atr, max_stop_pct))
                        target_pct_atr = max(stop_pct_atr * 1.5, min(target_pct_atr, max_stop_pct * 2))
                        
                        stop_price = entry_price - stop_distance
                        target_price = entry_price + target_distance
                    else:
                        # Fallback to fixed if ATR calculation fails
                        use_atr = False
            except Exception as e:
                logger.warning(f"[{pair}] ATR calculation failed: {e}, using fixed stops")
                use_atr = False
        
        # Use fixed percentage if ATR not available or disabled
        if not use_atr:
            stop_pct = stop_pct or 0.01  # Default 1%
            target_pct = target_pct or 0.02  # Default 2%
            stop_price = entry_price * (1 - stop_pct)
            target_price = entry_price * (1 + target_pct)
            stop_pct_atr = stop_pct
            target_pct_atr = target_pct
        
        # Calculate price change
        price_change_pct = (current_price - entry_price) / entry_price
        
        # Check stop-loss
        if current_price <= stop_price:
            return {
                "should_close": True,
                "reason": f"stop_loss_hit_{price_change_pct*100:.2f}%_atr" if use_atr else f"stop_loss_hit_{price_change_pct*100:.2f}%",
                "notional": current_notional,
                "entry_price": entry_price,
                "current_price": current_price,
                "price_change_pct": price_change_pct,
                "stop_price": stop_price,
                "target_price": target_price,
                "stop_pct": stop_pct_atr,
                "target_pct": target_pct_atr,
                "method": "atr" if use_atr else "fixed",
            }
        
        # Check take-profit
        if current_price >= target_price:
            return {
                "should_close": True,
                "reason": f"take_profit_hit_{price_change_pct*100:.2f}%_atr" if use_atr else f"take_profit_hit_{price_change_pct*100:.2f}%",
                "notional": current_notional,
                "entry_price": entry_price,
                "current_price": current_price,
                "price_change_pct": price_change_pct,
                "stop_price": stop_price,
                "target_price": target_price,
                "stop_pct": stop_pct_atr,
                "target_pct": target_pct_atr,
                "method": "atr" if use_atr else "fixed",
            }
        
        return {
            "should_close": False,
            "reason": f"holding_{price_change_pct*100:.2f}%",
            "notional": current_notional,
            "entry_price": entry_price,
            "current_price": current_price,
            "price_change_pct": price_change_pct,
            "stop_price": stop_price,
            "target_price": target_price,
            "stop_pct": stop_pct_atr,
            "target_pct": target_pct_atr,
            "method": "atr" if use_atr else "fixed",
        }



