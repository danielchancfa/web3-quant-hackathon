"""
Run inference for all configured pairs and write the outputs to disk.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config
from inference_service import InferenceService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model inference and store results.")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory with per-pair checkpoints.")
    parser.add_argument("--pairs", required=True, help="Comma-separated list of pairs, e.g. 'BTC/USD,ETH/USD'")
    parser.add_argument("--seq_daily", type=int, default=60)
    parser.add_argument("--seq_hourly", type=int, default=36)
    parser.add_argument("--seq_execution", type=int, default=36)
    parser.add_argument("--daily_label_mode", choices=["binary", "ternary"], default="binary")
    parser.add_argument("--output", type=str, default="artifacts/inference_output.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    db_path = Path(config.db_path)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]

    logger.info("Running inference for pairs: %s", ", ".join(pairs))
    service = InferenceService(
        db_path=db_path,
        checkpoint_dir=checkpoint_dir,
        seq_daily=args.seq_daily,
        seq_hourly=args.seq_hourly,
        seq_execution=args.seq_execution,
        daily_label_mode=args.daily_label_mode,
        pairs=pairs,
    )
    results = service.run_once()
    with output_path.open("w") as f:
        json.dump({"pairs": pairs, "results": results}, f, indent=2)

    logger.info("Inference written to %s", output_path)


if __name__ == "__main__":
    main()

