"""
Signal translation policy.

Takes raw model outputs and converts them into actionable trades:
- action: 'buy', 'sell', or 'hold'
- notional: dollar value to trade
- confidence: 0..1 confidence score
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class PolicyConfig:
    base_risk_fraction: float = 0.02  # 2% of portfolio per confident trade
    hourly_weight: float = 1.0
    execution_weight: float = 0.5
    min_signal: float = 0.003
    max_signal: float = 0.05
    default_regime_prob: float = 0.5
    min_confidence: float = 0.6
    # min_notional: Dict[str, float] = field(default_factory=lambda: {})
    min_trade_fraction: float = 0.01  # minimum 1% of portfolio per trade


def apply_policy(
    pair: str,
    inference_entry: Dict[str, Any],
    portfolio_value: float,
    position_limit_fraction: float,
    current_position_notional: float,
    config: PolicyConfig | None = None,
) -> Dict[str, Any]:
    """
    Convert inference outputs into order instructions.

    Returns dict with action, net_signal, confidence, notional.
    """
    config = config or PolicyConfig()
    daily = inference_entry["daily"]

    hourly_net = inference_entry["hourly"]["net_signal"]
    exec_net = inference_entry["execution"]["net_signal"]
    hold_conf_hour = inference_entry["hourly"]["raw_output"]["hold_confidence"]
    hold_conf_exec = inference_entry["execution"]["raw_output"]["hold_confidence"]

    # Combine hourly + execution signal
    net_combined = (
        config.hourly_weight * hourly_net +
        config.execution_weight * exec_net
    )

    # Regime weighting
    p_risk_on = daily["probabilities"].get("risk_on", config.default_regime_prob)
    net_weighted = net_combined * p_risk_on

    # Confidence (lower hold => higher confidence)
    confidence = max(0.0, 1.0 - max(hold_conf_hour, hold_conf_exec))

    signal_mag = abs(net_weighted)
    if signal_mag < config.min_signal or confidence < config.min_confidence:
        return {"pair": pair, "action": "hold", "net_signal": net_weighted, "confidence": confidence, "notional": 0.0}

    # Clamp signal magnitude
    signal_mag = min(signal_mag, config.max_signal)

    # Base notional budget
    base_notional = portfolio_value * config.base_risk_fraction
    notional = base_notional * signal_mag * confidence
    min_notional = portfolio_value * config.min_trade_fraction
    notional = max(notional, min_notional)

    # Respect position limits
    max_position = portfolio_value * position_limit_fraction
    if net_weighted > 0:  # buy
        remaining_limit = max(0.0, max_position - current_position_notional)
        notional = min(notional, remaining_limit)
    else:  # sell
        # avoid going short: limit sell amount to current position
        notional = min(notional, current_position_notional)

    if notional <= 0:
        return {"pair": pair, "action": "hold", "net_signal": net_weighted, "confidence": confidence, "notional": 0.0}

    action = "buy" if net_weighted > 0 else "sell"
    return {
        "pair": pair,
        "action": action,
        "net_signal": net_weighted,
        "confidence": confidence,
        "notional": notional,
    }


