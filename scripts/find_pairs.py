#!/usr/bin/env python3
"""
Find Pairs — Cron Job
Schedule: Weekly, Sunday 2 AM (0 2 * * 0)

Re-scans all asset combinations for cointegrated pairs.
Results are stored in DuckDB and logged.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manager import DataManager
from src.pairs.selector import PairSelector
from src.utils.config import load_config, get_all_tickers_flat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/find_pairs.log", mode="a"),
    ],
)
logger = logging.getLogger("find_pairs")


def main():
    start_time = datetime.now()
    logger.info(f"=== Pair scan started at {start_time} ===")

    try:
        cfg = load_config()
        dm = DataManager(cfg)
        selector = PairSelector(cfg)

        # Get all prices
        tickers = get_all_tickers_flat()
        prices = dm.get_prices(tickers)

        if prices.empty:
            logger.error("No price data available. Run data_refresh.py first.")
            sys.exit(1)

        # Drop tickers with too many NaNs
        min_obs = cfg["pair_selection"]["lookback_window"]
        valid = prices.columns[prices.count() >= min_obs]
        prices = prices[valid].dropna(how="all")

        logger.info(f"Scanning {len(valid)} assets with >={min_obs} observations")

        # Find pairs
        pairs_df = selector.find_pairs(prices)

        if pairs_df.empty:
            logger.warning("No valid pairs found")
        else:
            # Save to CSV for reference
            output_dir = Path("data/pairs")
            output_dir.mkdir(parents=True, exist_ok=True)
            date_str = datetime.now().strftime("%Y%m%d")
            pairs_df.to_csv(output_dir / f"pairs_{date_str}.csv", index=False)

            # Log top pairs
            logger.info(f"\nTop {len(pairs_df)} pairs:")
            for _, row in pairs_df.head(10).iterrows():
                logger.info(
                    f"  {row['pair_id']:30s} | corr={row['correlation']:.3f} "
                    f"| coint_p={row['coint_pvalue']:.4f} "
                    f"| half_life={row['half_life']:.1f}d "
                    f"| score={row['quality_score']:.3f}"
                )

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"=== Pair scan completed in {elapsed:.1f}s ===")

    except Exception as e:
        logger.error(f"Pair scan failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
