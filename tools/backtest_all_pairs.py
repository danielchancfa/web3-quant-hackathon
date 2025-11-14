# tools/backtest_all_pairs.py
from __future__ import annotations

import sys
from pathlib import Path
from subprocess import run, CalledProcessError, CompletedProcess
import csv
from datetime import datetime
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config



pairs = get_config().default_pairs
common_args = [
    "--db_path", "data_cache/trading_data.db",
    "--start", "2024-01-01T00:00:00",
    "--end", "2024-12-31T23:00:00",
    "--min_confidence", "0.55",
    "--min_edge", "0.0005",
    "--base_risk_fraction", "0.02",
    "--max_risk_fraction", "0.05",
]

log_path = PROJECT_ROOT / "logs" / "backtest_all_pairs.csv"
log_path.parent.mkdir(parents=True, exist_ok=True)
write_header = not log_path.exists()

with log_path.open("a", newline="") as csvfile:
    writer: csv.DictWriter[str] = csv.DictWriter(
        csvfile,
        fieldnames=[
            "timestamp",
            "pair",
            "status",
            "stdout",
            "stderr",
            "Equity Final [$]",
            "Return [%]",
            "Return (Ann.) [%]",
            "Volatility (Ann.) [%]",
            "Sharpe Ratio",
            "# Trades",
            "Win Rate [%]",
            "Max. Drawdown [%]",
        ],
    )
    if write_header:
        writer.writeheader()

    for pair in pairs:
        cmd = ["python", "-m", "tools.backtest_policy", "--pairs", pair, *common_args]
        print(f"\n=== Backtesting {pair} ===")
        try:
            proc: CompletedProcess = run(cmd, capture_output=True, text=True, check=True)
            stdout = proc.stdout.strip()
            print(stdout)
            summary: Dict[str, Any] = {
                "timestamp": datetime.utcnow().isoformat(),
                "pair": pair,
                "status": "success",
                "stdout": stdout,
                "stderr": proc.stderr.strip(),
            }
            for line in stdout.splitlines():
                if ": " in line:
                    key, value = line.split(": ", 1)
                    if key in writer.fieldnames:
                        summary[key] = value
            writer.writerow(summary)
        except CalledProcessError as exc:
            print(f"[WARN] Skipping {pair} (no forecasts or backtest error).")
            writer.writerow(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "pair": pair,
                    "status": "error",
                    "stdout": (exc.stdout or "").strip(),
                    "stderr": (exc.stderr or "").strip(),
                }
            )