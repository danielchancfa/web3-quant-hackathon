#!/usr/bin/env python3
"""
Iterative improvement loop:
1. Train model with current hyperparameters
2. Generate forecasts
3. Backfill actuals
4. Optimize policy thresholds
5. Run backtest
6. If results unsatisfactory, adjust hyperparameters and repeat
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and print it."""
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"[ERROR] {result.stderr}")
        sys.exit(1)
    return result


def main() -> None:
    pair = "ETH/USD"
    max_iterations = 3
    iteration = 1
    
    while iteration <= max_iterations:
        print(f"\n{'='*60}")
        print(f"=== Iteration {iteration}/{max_iterations} for {pair} ===")
        print(f"{'='*60}\n")
        
        # Step 1: Retrain model (use GPU for faster training)
        print(f"[1/5] Retraining {pair} on GPU...")
        run_cmd([
            "python", "-m", "prediction_model.train_next_bar",
            "--pair", pair,
            "--sequence_length", "256",
            "--epochs", "40",
            "--lr", "5e-5",
            "--confidence_weight", "0.3",
            "--device", "cuda",
            "--checkpoint_dir", "model_checkpoints/next_bar",
            "--db_path", "data_cache/trading_data.db",
        ])
        
        # Step 2: Generate forecasts (use GPU for faster inference)
        print(f"\n[2/5] Generating forecasts for {pair} on GPU...")
        run_cmd([
            "python", "-m", "tools.generate_historical_forecasts",
            "--pairs", pair,
            "--start", "2024-01-01T00:00:00",
            "--end", "2025-08-31T23:00:00",
            "--freq_hours", "1",
            "--device", "cuda",
            "--checkpoint_dir", "model_checkpoints/next_bar",
            "--db_path", "data_cache/trading_data.db",
        ])
        
        # Step 3: Backfill actuals
        print(f"\n[3/5] Backfilling actuals for {pair}...")
        run_cmd([
            "python", "-m", "tools.backfill_actuals",
            "--pairs", pair,
            "--db_path", "data_cache/trading_data.db",
        ])
        
        # Step 4: Optimize policy
        print(f"\n[4/5] Optimizing policy for {pair}...")
        opt_result = run_cmd([
            "python", "-m", "tools.optimize_policy",
            "--db_path", "data_cache/trading_data.db",
            "--pairs", pair,
            "--trials", "50",
            "--min_trades", "30",
            "--start", "2024-01-01T00:00:00",
            "--end", "2025-08-31T23:00:00",
        ])
        
        # Print optimization results
        if opt_result.stdout:
            print("\n" + "-"*60)
            print("OPTIMIZATION RESULTS:")
            print("-"*60)
            print(opt_result.stdout)
            print("-"*60)
        
        # Step 5: Run final backtest with default thresholds
        print(f"\n[5/5] Running final backtest for {pair}...")
        bt_result = run_cmd([
            "python", "-m", "tools.backtest_policy",
            "--pairs", pair,
            "--min_confidence", "0.50",
            "--min_edge", "0.0004",
            "--base_risk_fraction", "0.02",
            "--max_risk_fraction", "0.05",
            "--start", "2024-01-01T00:00:00",
            "--end", "2025-08-31T23:00:00",
            "--db_path", "data_cache/trading_data.db",
        ], check=False)
        
        # Print backtest results
        print("\n" + "="*60)
        print("BACKTEST RESULTS:")
        print("="*60)
        if bt_result.stdout:
            print(bt_result.stdout)
        if bt_result.stderr and bt_result.returncode != 0:
            print("ERRORS:")
            print(bt_result.stderr)
        print("="*60)
        
        # Also log to CSV (via backtest_all_pairs.py mechanism)
        print(f"\nResults also logged to: logs/backtest_all_pairs.csv")
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration} complete!")
        print(f"{'='*60}")
        print("\nKey metrics to check:")
        print("  - Sharpe Ratio (target: > 1.5)")
        print("  - # Trades (target: > 50)")
        print("  - Return [%] (target: > 10%)")
        print("  - Max. Drawdown [%] (target: < 15%)")
        print("\nIf results are unsatisfactory, adjust hyperparameters and continue.")
        print("Press Enter to continue to next iteration, or Ctrl+C to stop...")
        
        try:
            input()
        except KeyboardInterrupt:
            print("\nStopping improvement loop.")
            break
        
        iteration += 1
    
    print(f"\n{'='*60}")
    print("Improvement loop complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

