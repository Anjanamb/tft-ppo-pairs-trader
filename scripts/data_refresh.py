#!/usr/bin/env python3
"""
Data Refresh — Cron Job
Schedule: every 6 hours (0 */6 * * *)

Pulls latest market data from all configured sources
and stores it in DuckDB. This runs as a plain cron job
to conserve Claude Code routine quota.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manager import DataManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/data_refresh.log", mode="a"),
    ],
)
logger = logging.getLogger("data_refresh")


def main():
    start_time = datetime.now()
    logger.info(f"=== Data refresh started at {start_time} ===")

    try:
        dm = DataManager()
        dm.init_db()
        dm.refresh_all()

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"=== Data refresh completed in {elapsed:.1f}s ===")

    except Exception as e:
        logger.error(f"Data refresh failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
