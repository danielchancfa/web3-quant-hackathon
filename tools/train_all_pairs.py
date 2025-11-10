"""
Train separate models for each trading pair.

Usage:
    python tools/train_all_pairs.py --pairs BTC/USD,ETH/USD,...

Defaults to the competition's top pairs and uses the tuned hyperparameters.
"""

import argparse
import subprocess
from pathlib import Path

DEFAULT_PAIRS = [
    "BTC/USD",
    "ETH/USD",
    "BNB/USD",
    "SOL/USD",
    "XRP/USD",
    "ADA/USD",
    "DOGE/USD",
    "LTC/USD",
    "LINK/USD",
    "SUI/USD",
    "TRX/USD",
    "XLM/USD",
    "ZEC/USD",
]


def pair_to_dirname(pair: str) -> str:
    return pair.replace("/", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models for multiple trading pairs.")
    parser.add_argument("--pairs", type=str, default=",".join(DEFAULT_PAIRS),
                        help="Comma-separated list of pairs (e.g., BTC/USD,ETH/USD)")
    parser.add_argument("--epochs_daily", type=int, default=25)
    parser.add_argument("--epochs_hourly", type=int, default=30)
    parser.add_argument("--epochs_execution", type=int, default=15)
    parser.add_argument("--seq_daily", type=int, default=60)
    parser.add_argument("--seq_hourly", type=int, default=36)
    parser.add_argument("--seq_execution", type=int, default=36)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--daily_label_mode", type=str, choices=["binary", "ternary"], default="binary")
    parser.add_argument("--lr_hourly", type=float, default=3e-4)
    parser.add_argument("--lr_execution", type=float, default=1e-3)
    parser.add_argument("--lr_daily", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--batch_daily", type=int, default=32)
    parser.add_argument("--batch_hourly", type=int, default=64)
    parser.add_argument("--batch_execution", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]

    for pair in pairs:
        print(f"\n=== Training pair {pair} ===")
        out_dir = Path("model_checkpoints") / pair_to_dirname(pair)
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "-m", "modeling.train_transformer",
            "--epochs_daily", str(args.epochs_daily),
            "--epochs_hourly", str(args.epochs_hourly),
            "--epochs_execution", str(args.epochs_execution),
            "--seq_daily", str(args.seq_daily),
            "--seq_hourly", str(args.seq_hourly),
            "--seq_execution", str(args.seq_execution),
            "--dropout", str(args.dropout),
            "--daily_label_mode", args.daily_label_mode,
            "--lr_daily", str(args.lr_daily),
            "--lr_hourly", str(args.lr_hourly),
            "--lr_execution", str(args.lr_execution),
            "--val_ratio", str(args.val_ratio),
            "--batch_daily", str(args.batch_daily),
            "--batch_hourly", str(args.batch_hourly),
            "--batch_execution", str(args.batch_execution),
            "--pair", pair,
            "--output_dir", str(out_dir),
        ]

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

