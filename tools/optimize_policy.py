"""
Hyperparameter optimization for the trading policy using Optuna.

The script pulls historical forecasts (with realized next-bar outcomes) from
the `next_bar_forecasts` table, runs a simple backtest per trial, and
maximizes the annualized Sharpe ratio by tuning the policy parameters.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import optuna
import pandas as pd

from config import get_config
from prediction_execution import NextBarPolicyConfig, evaluate_forecast


def load_forecasts(
    db_path: Path,
    pairs: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load historical forecasts enriched with actual next-bar results."""
    where_clauses = ["actual_close IS NOT NULL"]
    params: List = []

    if pairs:
        placeholders = ",".join(["?"] * len(pairs))
        where_clauses.append(f"pair IN ({placeholders})")
        params.extend(pairs)

    if start:
        where_clauses.append("as_of >= ?")
        params.append(start)
    if end:
        where_clauses.append("as_of <= ?")
        params.append(end)

    query = f"""
        SELECT
            pair,
            as_of,
            reference_close,
            pred_open,
            pred_high,
            pred_low,
            pred_close,
            pred_volume,
            pred_delta_close,
            pred_delta_open,
            pred_delta_high,
            pred_delta_low,
            pred_delta_volume,
            confidence,
            actual_open,
            actual_high,
            actual_low,
            actual_close,
            actual_volume
        FROM next_bar_forecasts
        WHERE {' AND '.join(where_clauses)}
        ORDER BY as_of ASC
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(query, conn, params=params)

    if df.empty:
        raise RuntimeError("No historical forecasts with actual outcomes found for the given filters.")

    df["as_of"] = pd.to_datetime(df["as_of"])
    return df


def annualized_sharpe(returns: np.ndarray, freq_per_year: float = 24 * 365) -> float:
    """Compute annualized Sharpe ratio from a vector of periodic returns."""
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=1)
    if std_ret == 0:
        return float("-inf")
    scale = math.sqrt(freq_per_year)
    return mean_ret / std_ret * scale


def _safe_float(value) -> float:
    """Safely convert value to float, handling binary strings from corrupted DB."""
    if value is None:
        return 0.0
    if isinstance(value, bytes):
        # Binary data - try to decode as float32
        import struct
        try:
            return float(struct.unpack('f', value)[0])
        except (struct.error, ValueError):
            return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def build_forecast_row(row: pd.Series) -> Dict[str, object]:
    """Convert a DB row into the forecast dict expected by evaluate_forecast."""
    pred = {
        "close": _safe_float(row.pred_close),
        "confidence": _safe_float(row.confidence),
        "delta_close": _safe_float(row.pred_delta_close),
    }

    for key in ("open", "high", "low", "volume"):
        value = row[f"pred_{key}"]
        pred[key] = _safe_float(value) if value is not None else _safe_float(row.reference_close)

    for key in ("delta_open", "delta_high", "delta_low", "delta_volume"):
        value = row[f"pred_{key}"]
        pred[key] = _safe_float(value) if value is not None else 0.0

    forecast = {
        "pair": row.pair,
        "as_of": row.as_of.isoformat(),
        "reference_close": float(row.reference_close),
        "predicted": pred,
    }
    return forecast


def simulate_trades(
    data: pd.DataFrame,
    config: NextBarPolicyConfig,
) -> Dict[str, object]:
    """Run a single-pass backtest using the supplied policy configuration."""
    portfolio_value = 1.0  # normalized capital
    returns: List[float] = []
    trade_actions: List[int] = []  # +1 buy, -1 sell, 0 hold

    for _, row in data.iterrows():
        forecast = build_forecast_row(row)
        instruction = evaluate_forecast(
            forecast=forecast,
            portfolio_value=portfolio_value,
            current_position_notional=0.0,
            config=config,
        )

        actual_close = row.actual_close
        if actual_close is None:
            returns.append(0.0)
            trade_actions.append(0)
            continue

        ref_close = row.reference_close
        price_change = (actual_close - ref_close) / ref_close
        notional = float(instruction.get("notional", 0.0))

        if instruction["action"] == "buy" and notional > 0:
            returns.append(notional * price_change)
            trade_actions.append(1)
        elif instruction["action"] == "sell" and notional > 0:
            # In practice we rarely hit this branch because we assume flat positions,
            # but we include it for completeness.
            returns.append(-notional * price_change)
            trade_actions.append(-1)
        else:
            returns.append(0.0)
            trade_actions.append(0)

    returns_array = np.array(returns, dtype=float)
    trades = sum(1 for a in trade_actions if a != 0)
    sharpe = annualized_sharpe(returns_array) if trades > 0 else float("-inf")

    stats = {
        "sharpe": sharpe,
        "trades": trades,
        "mean_return": returns_array.mean(),
        "std_return": returns_array.std(ddof=1) if trades > 1 else 0.0,
        "returns": returns_array,
        "actions": trade_actions,
    }
    return stats


def objective_factory(
    data: pd.DataFrame,
    min_trades: int,
) -> optuna.study.Study:
    """Create the Optuna objective function capturing the data context."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "base_risk_fraction": trial.suggest_float("base_risk_fraction", 0.005, 0.05, log=True),
            "max_risk_fraction": trial.suggest_float("max_risk_fraction", 0.01, 0.25),
            "min_confidence": trial.suggest_float("min_confidence", 0.4, 0.8),
            "min_edge": trial.suggest_float("min_edge", 1e-4, 0.003, log=True),
            "target_buffer": trial.suggest_float("target_buffer", 0.0, 0.5),
            "stop_buffer": trial.suggest_float("stop_buffer", 0.0, 0.5),
        }

        if params["max_risk_fraction"] < params["base_risk_fraction"]:
            raise optuna.TrialPruned()

        cfg = NextBarPolicyConfig(
            base_risk_fraction=params["base_risk_fraction"],
            max_risk_fraction=params["max_risk_fraction"],
            min_confidence=params["min_confidence"],
            min_edge=params["min_edge"],
            target_buffer=params["target_buffer"],
            stop_buffer=params["stop_buffer"],
            allow_shorts=False,
            force_flatten=True,
        )

        stats = simulate_trades(data, cfg)
        if stats["trades"] < min_trades or not math.isfinite(stats["sharpe"]):
            raise optuna.TrialPruned()

        trial.set_user_attr("trades", stats["trades"])
        trial.set_user_attr("mean_return", stats["mean_return"])
        trial.set_user_attr("std_return", stats["std_return"])
        return stats["sharpe"]

    return objective


def run_backtest_with_params(
    data: pd.DataFrame,
    params: Dict[str, float],
) -> Dict[str, object]:
    cfg = NextBarPolicyConfig(
        base_risk_fraction=params["base_risk_fraction"],
        max_risk_fraction=params["max_risk_fraction"],
        min_confidence=params["min_confidence"],
        min_edge=params["min_edge"],
        target_buffer=params["target_buffer"],
        stop_buffer=params["stop_buffer"],
        allow_shorts=False,
        force_flatten=True,
    )
    return simulate_trades(data, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize trading policy parameters for maximum Sharpe ratio.")
    parser.add_argument("--db_path", type=Path, default=None, help="Path to SQLite database (defaults to config).")
    parser.add_argument("--pairs", type=str, default="", help="Comma-separated list of pairs to include.")
    parser.add_argument("--start", type=str, default=None, help="Start timestamp (ISO 8601).")
    parser.add_argument("--end", type=str, default=None, help="End timestamp (ISO 8601).")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--min_trades", type=int, default=10, help="Minimum number of trades required per trial.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    config = get_config()
    db_path = args.db_path or Path(config.db_path)

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()] if args.pairs else None
    data = load_forecasts(db_path=db_path, pairs=pairs, start=args.start, end=args.end)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    objective = objective_factory(data=data, min_trades=args.min_trades)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print("\n=== Optimization Results ===")
    print(f"Best Sharpe: {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best_stats = run_backtest_with_params(data, study.best_params)
    print(f"Trades executed: {best_stats['trades']}")
    print(f"Mean return: {best_stats['mean_return']:.6f}")
    print(f"Std return: {best_stats['std_return']:.6f}")

    # Optional: display a summary of non-zero trades
    returns = best_stats["returns"]
    actions = best_stats["actions"]
    executed = [(r, a) for r, a in zip(returns, actions) if a != 0]
    if executed:
        print("\nSample of executed trade returns (first 10):")
        for idx, (ret, action) in enumerate(executed[:10], 1):
            side = "BUY" if action > 0 else "SELL"
            print(f"  #{idx:02d} {side}: return={ret:.6f}")


if __name__ == "__main__":
    main()

