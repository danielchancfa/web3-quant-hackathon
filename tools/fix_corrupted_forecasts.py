#!/usr/bin/env python3
"""
Fix corrupted forecast data by deleting and regenerating forecasts for specified pairs.
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config


def delete_forecasts(db_path: Path, pairs: list[str]) -> None:
    """Delete forecasts for specified pairs."""
    with sqlite3.connect(db_path) as conn:
        placeholders = ",".join(["?"] * len(pairs))
        conn.execute(
            f"DELETE FROM next_bar_forecasts WHERE pair IN ({placeholders})",
            pairs,
        )
        conn.commit()
        print(f"Deleted forecasts for {len(pairs)} pairs: {pairs}")


def regenerate_forecasts(
    db_path: Path,
    pairs: list[str],
    checkpoint_dir: Path,
    start: str,
    end: str,
    device: str = "cuda",
) -> None:
    """Regenerate forecasts for specified pairs."""
    for pair in pairs:
        print(f"\nRegenerating forecasts for {pair}...")
        subprocess.run(
            [
                "python",
                "-m",
                "tools.generate_historical_forecasts",
                "--pairs",
                pair,
                "--start",
                start,
                "--end",
                end,
                "--freq_hours",
                "1",
                "--device",
                device,
                "--checkpoint_dir",
                str(checkpoint_dir),
                "--db_path",
                str(db_path),
            ],
            check=True,
        )

        subprocess.run(
            [
                "python",
                "-m",
                "tools.backfill_actuals",
                "--pairs",
                pair,
                "--db_path",
                str(db_path),
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix corrupted forecast data.")
    parser.add_argument("--pairs", type=str, required=True, help="Comma-separated list of pairs to fix.")
    parser.add_argument("--db_path", type=Path, default=None, help="Path to SQLite database.")
    parser.add_argument("--checkpoint_dir", type=Path, default="model_checkpoints/next_bar", help="Checkpoint directory.")
    parser.add_argument("--start", type=str, default="2024-01-01T00:00:00", help="Start timestamp.")
    parser.add_argument("--end", type=str, default="2024-12-31T23:00:00", help="End timestamp.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu).")
    args = parser.parse_args()

    config = get_config()
    db_path = args.db_path or Path(config.db_path)
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]

    print(f"Fixing forecasts for: {pairs}")
    print("Step 1: Deleting corrupted forecasts...")
    delete_forecasts(db_path, pairs)

    print("\nStep 2: Regenerating forecasts...")
    regenerate_forecasts(
        db_path=db_path,
        pairs=pairs,
        checkpoint_dir=args.checkpoint_dir,
        start=args.start,
        end=args.end,
        device=args.device,
    )

    print("\n✅ Forecasts fixed!")


if __name__ == "__main__":
    main()

