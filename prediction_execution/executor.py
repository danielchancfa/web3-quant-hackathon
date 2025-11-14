"""
Execution helpers for next-bar forecast instructions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

from execution.executor import execute_trade, ExecutionConfig, get_last_price
from execution.position_manager import PositionManager

logger = logging.getLogger(__name__)


@dataclass
class ForecastInstruction:
    pair: str
    action: str
    notional: float
    confidence: float
    edge: float
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    exit_time: Optional[str] = None
    reason: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForecastInstruction":
        return cls(
            pair=data["pair"],
            action=data.get("action", "hold"),
            notional=float(data.get("notional", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            edge=float(data.get("edge", 0.0)),
            target_price=(None if data.get("target_price") is None else float(data["target_price"])),
            stop_price=(None if data.get("stop_price") is None else float(data["stop_price"])),
            exit_time=data.get("exit_time"),
            reason=data.get("reason", ""),
        )


class ForecastExecutor:
    """
    Executes forecast-based instructions using the existing trading infrastructure.
    """

    def __init__(
        self,
        position_manager: PositionManager,
        execution_config: ExecutionConfig | None = None,
        paper: bool = False,
    ) -> None:
        self.position_manager = position_manager
        self.execution_config = execution_config or ExecutionConfig()
        self.paper = paper

    def execute(self, instruction: Dict[str, Any] | ForecastInstruction) -> Dict[str, Any]:
        instr = instruction if isinstance(instruction, ForecastInstruction) else ForecastInstruction.from_dict(instruction)
        pair = instr.pair

        if instr.action == "hold" or instr.notional <= 0:
            logger.info(f"[{pair}] Policy returned hold ({instr.reason})")
            return {
                "status": "hold",
                "pair": pair,
                "reason": instr.reason,
                "instruction": instr,
            }

        side = "buy" if instr.action.lower() == "buy" else "sell"

        if self.paper:
            price = get_last_price(pair)
            self.position_manager.apply_simulated(pair, side, instr.notional, price)
            logger.info(
                f"[{pair}] Paper {side.upper()} notional ${instr.notional:.2f} @ {price:.4f} "
                f"(conf={instr.confidence:.2f}, edge={instr.edge:.5f}, reason={instr.reason})"
            )
            return {
                "status": "paper",
                "pair": pair,
                "side": side.upper(),
                "price": price,
                "notional": instr.notional,
                "instruction": instr,
            }

        # TODO: Implement target/stop order placement if Roostoo API supports it
        # For now, we place market orders. Target/stop logic is used in backtesting.
        # If instr.target_price and instr.stop_price are set, they could be used for
        # limit orders or stop-loss/take-profit orders if the exchange supports them.
        
        result = execute_trade(pair, side, instr.notional, self.execution_config)
        status = result.get("status", "submitted")
        price = float(result.get("price", get_last_price(pair)))
        actual_notional = float(result.get("notional", instr.notional))
        self.position_manager.apply_trade(pair, side.upper(), actual_notional, price)

        logger.info(
            f"[{pair}] Live {side.upper()} status={status} notional=${actual_notional:.2f} "
            f"price={price:.4f} conf={instr.confidence:.2f} edge={instr.edge:.5f} reason={instr.reason}"
        )
        if instr.target_price or instr.stop_price:
            logger.info(
                f"[{pair}] Target/Stop prices: target={instr.target_price}, stop={instr.stop_price} "
                f"(not yet implemented for live trading)"
            )

        return {
            "status": status,
            "pair": pair,
            "side": side.upper(),
            "price": price,
            "notional": actual_notional,
            "api_response": result,
            "instruction": instr,
        }

