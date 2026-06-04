#!/usr/bin/env python3
"""
Train the TFT spread predictor.

Builds the multi-pair feature panel from the pairs discovered by
``find_pairs.py``, fits a single Temporal Fusion Transformer across all of them,
and saves the best checkpoint to ``models/``. With ``--interpret`` it also logs
the TFT variable importances — which inputs the model actually leans on.

Usage:
    python scripts/train_tft.py                      # all pairs, config epochs
    python scripts/train_tft.py --top 5 --epochs 30  # quick run on top 5 pairs
    python scripts/train_tft.py --interpret
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manager import DataManager
from src.models.dataset import TFTDatasetBuilder
from src.models.tft_predictor import TFTPredictor
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        # utf-8 so third-party log lines with emoji don't crash on Windows cp1252
        logging.FileHandler("logs/train_tft.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_tft")


def _latest_pairs_file() -> Path:
    """Most recent pairs_*.csv produced by find_pairs.py."""
    files = sorted(Path("data/pairs").glob("pairs_*.csv"))
    if not files:
        logger.error("No pairs_*.csv found. Run scripts/find_pairs.py first.")
        sys.exit(1)
    return files[-1]


def main():
    parser = argparse.ArgumentParser(description="Train the TFT spread predictor.")
    parser.add_argument("--pairs", type=Path, default=None,
                        help="pairs_*.csv to use (default: latest in data/pairs)")
    parser.add_argument("--top", type=int, default=None,
                        help="train on the top N pairs by quality score")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override tft.max_epochs from config")
    parser.add_argument("--output", type=Path, default=None,
                        help="checkpoint path (default: models/tft_<date>.ckpt)")
    parser.add_argument("--interpret", action="store_true",
                        help="log variable importances after training")
    args = parser.parse_args()

    from src.utils.runtime import configure_quiet_runtime
    configure_quiet_runtime()

    start = datetime.now()
    logger.info("=== TFT training started at %s ===", start)

    cfg = load_config()
    if args.epochs is not None:
        cfg["tft"]["max_epochs"] = args.epochs

    pairs_file = args.pairs or _latest_pairs_file()
    pairs_df = pd.read_csv(pairs_file)
    if args.top:
        pairs_df = pairs_df.head(args.top)
    logger.info("Using %d pairs from %s", len(pairs_df), pairs_file.name)

    builder = TFTDatasetBuilder(cfg)
    panel = builder.build_panel(pairs_df, DataManager(cfg))
    if panel.empty:
        logger.error("Empty feature panel — is the database populated?")
        sys.exit(1)

    training, validation = builder.make_datasets(panel)

    predictor = TFTPredictor(cfg)
    metrics = predictor.train(training, validation)
    logger.info(
        "Trained in %d epochs — best val_loss=%s",
        metrics["epochs_run"], metrics["best_val_loss"],
    )

    output = args.output or Path("models") / f"tft_{start:%Y%m%d}.ckpt"
    predictor.save(output)
    logger.info("Checkpoint saved to %s", output)

    if args.interpret:
        _log_importances(predictor, validation)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info("=== TFT training completed in %.1fs ===", elapsed)


def _log_importances(predictor: TFTPredictor, validation):
    """Log the TFT's encoder/decoder variable importances (top inputs)."""
    interp = predictor.interpret(validation)
    model = predictor.model

    for kind, names in (
        ("encoder", model.encoder_variables),
        ("decoder", model.decoder_variables),
    ):
        importance = interp[f"{kind}_variables"].detach().cpu().numpy()
        importance = importance / (importance.sum() + 1e-8)
        ranked = sorted(zip(names, importance), key=lambda x: -x[1])
        logger.info("Top %s variables:", kind)
        for name, share in ranked[:8]:
            logger.info("  %-24s %5.1f%%", name, 100 * share)


if __name__ == "__main__":
    main()
