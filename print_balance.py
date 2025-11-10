"""
Utility script to fetch and print Roostoo account balances.
"""

from __future__ import annotations

import json
from execution.position_manager import PositionManager
from config import get_config


def main() -> None:
    config = get_config()
    pairs = ["BTC/USD", "ETH/USD", "BNB/USD", "SOL/USD", "XRP/USD"]
    manager = PositionManager(pairs)
    manager.refresh()

    snapshot = {
        "cash_usd": manager.cash_usd,
        "asset_qty": manager.asset_qty,
        "pair_notionals": manager.pair_notionals,
        "total_value": manager.total_value(),
    }
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()

