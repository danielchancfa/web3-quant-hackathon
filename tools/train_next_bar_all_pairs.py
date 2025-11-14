"""
Train the next-bar transformer for a list of trading pairs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from config import get_config

DEFAULT_PAIRS = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "BNB/USD",
    "XRP/USD",
    "ADA/USD",
    "DOGE/USD",
    "TRX/USD",
    "LTC/USD",
    "LINK/USD",
    "HBAR/USD",
    "XLM/USD",
    "ZEC/USD",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train next-bar models for multiple pairs.")
    parser.add_argument(
        "--pairs",
        type=str,
        default=",".join(DEFAULT_PAIRS),
        help="Comma-separated list of pairs (default: 13 pairs used in the project).",
    )
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--confidence_weight", type=float, default=0.5)
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("model_checkpoints/next_bar"),
    )
    parser.add_argument("--db_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None, help="Force device (overrides per-run default).")
    parser.add_argument("--extra_args", type=str, default="", help="Additional args passed through verbatim.")
    return parser.parse_args()


def build_command(
    pair: str,
    args: argparse.Namespace,
    config_db: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "prediction_model.train_next_bar",
        "--pair",
        pair,
        "--sequence_length",
        str(args.sequence_length),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--confidence_weight",
        str(args.confidence_weight),
        "--checkpoint_dir",
        str(args.checkpoint_dir),
        "--db_path",
        str(config_db),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.extra_args:
        cmd.extend(args.extra_args.split())
    return cmd


def main() -> None:
    args = parse_args()
    config = get_config()
    db_path = args.db_path or getattr(config, "sqlite_path", None) or Path("data_cache/trading_data.db")

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    if not pairs:
        raise ValueError("No trading pairs provided.")

    print(f"Training next-bar models for {len(pairs)} pairs...")
    for idx, pair in enumerate(pairs, start=1):
        cmd = build_command(pair, args, Path(db_path))
        print(f"[{idx}/{len(pairs)}] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    print("All runs completed.")


if __name__ == "__main__":
    main()

