#!/usr/bin/env python3
"""
Walk-forward backtest — Cron Job
Schedule: Weekly, Sunday 3 AM (after the pair scan).

Runs a rolling train/test backtest on the top pair and reports out-of-sample
performance against a buy-and-hold benchmark. The hedge ratio is re-estimated
every fold, so no future information leaks into the spread.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --strategy ppo --timesteps 50000
    python scripts/run_backtest.py --pair GC=F__GS --strategy both
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import WalkForwardBacktester, benchmark_metrics
from src.backtest.strategies import (
    ppo_strategy,
    regime_zscore_strategy,
    zscore_strategy,
)
from src.data.manager import DataManager
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/run_backtest.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_backtest")

_REPORT_KEYS = [
    "annualized_return", "sharpe_ratio", "sortino_ratio",
    "max_drawdown", "calmar_ratio", "win_rate", "profit_factor",
]


def _log_metrics(label: str, m: dict):
    logger.info("--- %s ---", label)
    for k in _REPORT_KEYS:
        if k in m:
            logger.info("  %-20s %8.3f", k, m[k])


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest.")
    parser.add_argument("--pair", type=str, default=None)
    parser.add_argument("--strategy", choices=["zscore", "ppo", "both"],
                        default="both")
    parser.add_argument("--timesteps", type=int, default=30000,
                        help="PPO timesteps per fold")
    args = parser.parse_args()

    start = datetime.now()
    logger.info("=== Backtest started at %s ===", start)

    cfg = load_config()
    pairs_df = pd.read_csv(sorted(Path("data/pairs").glob("pairs_*.csv"))[-1])
    row = (
        pairs_df[pairs_df["pair_id"] == args.pair].iloc[0]
        if args.pair else pairs_df.iloc[0]
    )
    ticker_a, ticker_b = row["ticker_a"], row["ticker_b"]
    logger.info("Backtesting %s (%s vs %s)", row["pair_id"], ticker_a, ticker_b)

    dm = DataManager(cfg)
    prices = dm.get_prices([ticker_a, ticker_b]).dropna()
    if prices.empty:
        logger.error("No price data — run data_refresh.py first.")
        sys.exit(1)

    bt = WalkForwardBacktester(cfg)
    strategies = {}
    if args.strategy in ("zscore", "both"):
        strategies["z-score rule"] = zscore_strategy()
        strategies["z-score + regime"] = regime_zscore_strategy()
    if args.strategy in ("ppo", "both"):
        strategies["PPO (refit/fold)"] = ppo_strategy(timesteps=args.timesteps)

    bench_ticker = cfg["backtest"]["benchmark"]
    bench_close = dm.get_prices([bench_ticker]).get(bench_ticker)

    for label, strat in strategies.items():
        result = bt.run(prices[ticker_a], prices[ticker_b], strat)
        logger.info("%s: %d folds, %d OOS days",
                    label, result["n_folds"], len(result["returns"]))
        _log_metrics(f"{label} (out-of-sample, stitched)", result["metrics"])
        if bench_close is not None:
            bench = benchmark_metrics(bench_close, result["dates"])
            if bench:
                _log_metrics(f"{bench_ticker} buy-and-hold (same dates)", bench)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info("=== Backtest completed in %.1fs ===", elapsed)


if __name__ == "__main__":
    main()
