"""
Cron-friendly wrapper to refresh all datasets in the SQLite database.
"""

from __future__ import annotations

import logging

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.historical_data_pipeline import main as pipeline_main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting data pipeline refresh.")
    pipeline_main()
    logger.info("Data pipeline refresh complete.")


if __name__ == "__main__":
    main()

