"""
Export Horus market price history to CSV files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from config import get_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Horus hourly price history to CSV.")
    parser.add_argument(
        "--assets",
        type=str,
        default="",
        help="Comma-separated list of assets (e.g. BTC,ETH). Default: all assets in horus_market_price.",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        choices=("15m", "1h", "1d"),
        help="Interval to export (default: 1h).",
    )
    parser.add_argument(
        "--db_path",
        type=Path,
        default=None,
        help="Override SQLite database path.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data_exports/horus_prices"),
        help="Directory to write CSV files into.",
    )
    return parser.parse_args()


def get_assets(conn: "sqlite3.Connection", interval: str, requested: Iterable[str] | None = None) -> List[str]:
    df = pd.read_sql(
        "SELECT DISTINCT asset FROM horus_market_price WHERE interval = ? ORDER BY asset",
        conn,
        params=(interval,),
    )
    assets = df["asset"].tolist()
    if requested:
        requested_set = {a.upper() for a in requested}
        assets = [a for a in assets if a.upper() in requested_set]
    return assets


def main() -> None:
    args = parse_args()
    config = get_config()
    db_path = args.db_path or getattr(config, "sqlite_path", None) or Path("data_cache/trading_data.db")
    db_path = Path(db_path)

    assets_requested = [s.strip() for s in args.assets.split(",") if s.strip()] or None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        assets = get_assets(conn, args.interval, assets_requested)
        if not assets:
            raise RuntimeError(f"No assets found for interval {args.interval}.")

        for asset in assets:
            df = pd.read_sql(
                """
                SELECT datetime, price
                FROM horus_market_price
                WHERE asset = ? AND interval = ?
                ORDER BY datetime
                """,
                conn,
                params=(asset, args.interval),
            )
            if df.empty:
                print(f"[WARN] No data for {asset} ({args.interval}). Skipping.")
                continue
            df["datetime"] = pd.to_datetime(df["datetime"])
            output_path = args.output_dir / f"{asset.lower()}_{args.interval}.csv"
            df.to_csv(output_path, index=False)
            print(f"[OK] Exported {len(df)} rows for {asset} ({args.interval}) -> {output_path}")


if __name__ == "__main__":
    main()

