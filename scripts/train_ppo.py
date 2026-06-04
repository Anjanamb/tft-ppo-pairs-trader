#!/usr/bin/env python3
"""
Train the PPO trading agent on TFT spread forecasts.

Pipeline: build one pair's feature panel -> roll the TFT 1-step forecasts across
it (if a checkpoint is given) -> feed spread + forecast edge + uncertainty into
the trading env -> train the agent -> score it against flat / random / z-score
baselines.

Usage:
    python scripts/train_ppo.py --tft models/tft_20260604.ckpt
    python scripts/train_ppo.py --pair BNB/USDT__XLF --timesteps 200000
    python scripts/train_ppo.py            # no TFT: trades on raw spread features
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.evaluation import (
    build_env_inputs,
    evaluate,
    flat_policy,
    random_policy,
    zscore_policy,
)
from src.agents.ppo_agent import TradingAgent
from src.agents.trading_env import PairsTradingEnv
from src.data.manager import DataManager
from src.models.dataset import TFTDatasetBuilder
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train_ppo.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_ppo")


def _latest_pairs_file() -> Path:
    files = sorted(Path("data/pairs").glob("pairs_*.csv"))
    if not files:
        logger.error("No pairs_*.csv found. Run scripts/find_pairs.py first.")
        sys.exit(1)
    return files[-1]


def _forecasts(tft_path: Path, panel: pd.DataFrame, cfg: dict):
    """Rolling 1-step TFT forecasts for the panel, or None if no checkpoint."""
    if tft_path is None:
        logger.info("No TFT checkpoint — agent trades on raw spread features only")
        return None
    from src.models.tft_predictor import TFTPredictor

    predictor = TFTPredictor(cfg)
    predictor.load(tft_path)
    forecasts = predictor.predict_per_step(panel)
    logger.info("Generated %d rolling forecasts", len(forecasts))
    return forecasts


def main():
    parser = argparse.ArgumentParser(description="Train the PPO trading agent.")
    parser.add_argument("--pair", type=str, default=None,
                        help="pair_id to trade (default: top pair by score)")
    parser.add_argument("--tft", type=Path, default=None,
                        help="TFT checkpoint for forecast features")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="override ppo.total_timesteps")
    parser.add_argument("--algo", type=str, default=None,
                        help="override ppo.algorithm (PPO, A2C, ...)")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="chronological holdout for out-of-sample scoring")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    start = datetime.now()
    logger.info("=== PPO training started at %s ===", start)

    cfg = load_config()
    if args.algo:
        cfg["ppo"]["algorithm"] = args.algo

    pairs_df = pd.read_csv(_latest_pairs_file())
    row = (
        pairs_df[pairs_df["pair_id"] == args.pair]
        if args.pair
        else pairs_df.head(1)
    )
    if row.empty:
        logger.error("Pair %s not in pairs file", args.pair)
        sys.exit(1)
    pair_id = row.iloc[0]["pair_id"]
    logger.info("Trading pair: %s", pair_id)

    panel = TFTDatasetBuilder(cfg).build_panel(row, DataManager(cfg))
    if panel.empty:
        logger.error("Empty panel for %s", pair_id)
        sys.exit(1)

    forecasts = _forecasts(args.tft, panel, cfg)
    spread, forecast, uncertainty = build_env_inputs(panel, forecasts)

    # Chronological train/test split. The agent only ever sees the train window;
    # the test window is held out so the reported edge is not just memorization.
    # (A full rolling walk-forward comes in Phase 7 — this is the honest minimum.)
    split = int(len(spread) * (1 - args.test_fraction))
    train_env = PairsTradingEnv(
        spread[:split], forecast[:split], uncertainty[:split], config=cfg
    )
    test_env = PairsTradingEnv(
        spread[split:], forecast[split:], uncertainty[split:], config=cfg
    )
    logger.info("Episode: %d train / %d test steps", split, len(spread) - split)

    agent = TradingAgent(cfg).train(
        train_env, total_timesteps=args.timesteps, seed=cfg["tft"]["seed"]
    )
    output = args.output or Path("models") / f"ppo_{start:%Y%m%d}.zip"
    agent.save(output)

    scaling = cfg["ppo"]["reward_scaling"]

    def score(env, label: str):
        policies = {
            "PPO": lambda obs: agent.predict(obs, deterministic=True),
            "z-score rule": zscore_policy(),
            "random": random_policy(env),
            "flat": flat_policy,
        }
        logger.info("--- %s ---", label)
        logger.info("%-14s %8s %8s %8s %7s",
                    "policy", "PnL", "Sharpe", "maxDD", "trades")
        for name, fn in policies.items():
            m = evaluate(env, fn, scaling)
            logger.info("%-14s %8.4f %8.2f %8.4f %7d", name,
                        m["total_pnl"], m["sharpe"], m["max_drawdown"], m["n_trades"])

    score(train_env, "IN-SAMPLE (train window - expect optimism)")
    score(test_env, "OUT-OF-SAMPLE (held-out test window)")

    elapsed = (datetime.now() - start).total_seconds()
    logger.info("=== PPO training completed in %.1fs ===", elapsed)


if __name__ == "__main__":
    main()
