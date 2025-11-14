"""
Populate actual next-bar outcomes in next_bar_forecasts from the OHLCV table.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import get_config


def floor_hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)


def backfill(
    db_path: Path,
    pairs: Optional[List[str]],
    start: Optional[str],
    end: Optional[str],
) -> int:
    with sqlite3.connect(db_path) as conn:
        forecasts = pd.read_sql_query(
            """
            SELECT id, pair, as_of
            FROM next_bar_forecasts
            WHERE actual_close IS NULL
            """,
            conn,
        )
        if forecasts.empty:
            return 0

        forecasts["as_of"] = pd.to_datetime(
            forecasts["as_of"],
            utc=True,
            format="ISO8601",
            errors="coerce",
        )
        forecasts = forecasts.dropna(subset=["as_of"])

        if pairs:
            forecasts = forecasts[forecasts["pair"].isin(pairs)]
        if start:
            start_ts = pd.to_datetime(start)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            forecasts = forecasts[forecasts["as_of"] >= start_ts]
        if end:
            end_ts = pd.to_datetime(end)
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            forecasts = forecasts[forecasts["as_of"] <= end_ts]

        if forecasts.empty:
            return 0

        updates = 0
        for _, row in forecasts.iterrows():
            as_of = pd.to_datetime(row["as_of"], utc=True)
            next_time = floor_hour(as_of) + timedelta(hours=1)
            lookup_time = next_time.tz_convert(None)
            pair = row["pair"]

            ohlcv = pd.read_sql_query(
                """
                SELECT datetime, open, high, low, close, volume
                FROM ohlcv
                WHERE pair = ?
                  AND interval = '1h'
                  AND datetime = ?
                """,
                conn,
                params=(pair, lookup_time.isoformat()),
            )

            if ohlcv.empty:
                continue

            rec = ohlcv.iloc[0]
            conn.execute(
                """
                UPDATE next_bar_forecasts
                SET actual_open = ?,
                    actual_high = ?,
                    actual_low = ?,
                    actual_close = ?,
                    actual_volume = ?
                WHERE id = ?
                """,
                (
                    rec["open"],
                    rec["high"],
                    rec["low"],
                    rec["close"],
                    rec["volume"],
                    int(row["id"]),
                ),
            )
            updates += 1

        conn.commit()
        return updates


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill actual OHLCV values into next_bar_forecasts.")
    parser.add_argument("--db_path", type=Path, default=None, help="Path to SQLite database.")
    parser.add_argument("--pairs", type=str, default="", help="Comma-separated list of pairs to update.")
    parser.add_argument("--start", type=str, default=None, help="Start timestamp (ISO 8601).")
    parser.add_argument("--end", type=str, default=None, help="End timestamp (ISO 8601).")
    args = parser.parse_args()

    config = get_config()
    db_path = args.db_path or Path(config.db_path)
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()] if args.pairs else None

    updated = backfill(db_path=db_path, pairs=pairs, start=args.start, end=args.end)
    print(f"Updated {updated} forecast rows with actual OHLCV.")


if __name__ == "__main__":
    main()

