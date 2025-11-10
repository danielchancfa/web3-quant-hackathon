"""
Utility script to inspect daily coverage in the SQLite database.
"""

import sqlite3
from pathlib import Path

from config import get_config


def main() -> None:
    db_path = Path(get_config().db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print("Daily row counts per pair:")
    cur.execute(
        """
        SELECT pair, COUNT(*), MIN(datetime), MAX(datetime)
        FROM ohlcv
        WHERE interval='1d'
        GROUP BY pair
        ORDER BY pair
        """
    )
    for pair, count, start, end in cur.fetchall():
        print(f"  {pair}: rows={count}, range={start} -> {end}")

    print("\nOther daily tables:")
    for table in [
        "horus_transaction_count",
        "horus_chain_tvl",
        "coinmarketcap_fear_greed",
    ]:
        cur.execute(f"SELECT COUNT(*), MIN(datetime), MAX(datetime) FROM {table}")
        count, start, end = cur.fetchone()
        print(f"  {table}: rows={count}, range={start} -> {end}")

    conn.close()


if __name__ == "__main__":
    main()

