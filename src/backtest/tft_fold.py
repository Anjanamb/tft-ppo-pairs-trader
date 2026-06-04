"""
Per-fold TFT forecaster for the walk-forward backtest.

This is the look-ahead-free way to put the TFT inside the backtest: for each
fold a *fresh* (small, short-trained) TFT is fit on the train window only — using
that fold's train-estimated hedge ratio — and asked for rolling 1-step spread
forecasts. The forecasts are returned indexed by date so the engine can align
them to its held-out test window without any index bookkeeping.

It is deliberately a reduced model (smaller width, fewer epochs): the point is an
honest signal per fold, fast enough to run 10+ folds, not a state-of-the-art fit.
"""

import copy
import logging

import pandas as pd

from src.models.dataset import TFTDatasetBuilder
from src.models.features import SpreadFeatureEngineer
from src.models.tft_predictor import TFTPredictor

logger = logging.getLogger(__name__)


def _fold_config(cfg: dict, epochs: int) -> dict:
    c = copy.deepcopy(cfg)
    c["tft"].update(
        max_epochs=epochs,
        hidden_size=32,
        hidden_continuous_size=16,
        attention_head_size=2,
        max_encoder_length=30,
        max_prediction_length=5,
        early_stopping_patience=4,
    )
    return c


def forecast_fold(
    cfg: dict,
    close_a: pd.Series,
    close_b: pd.Series,
    beta: float,
    train_len: int,
    epochs: int = 20,
) -> pd.DataFrame:
    """
    Train a TFT on the fold's train window and forecast the whole fold span.

    Args:
        cfg: base config (a reduced TFT config is derived from it).
        close_a, close_b: aligned closes for the fold span [start : test_end].
        beta: hedge ratio estimated on the train window only.
        train_len: number of leading rows that belong to the train window.
        epochs: training budget per fold.

    Returns:
        DataFrame indexed by datetime with ``prediction`` and ``uncertainty``
        (q90-q10) columns. Empty if the fold is too short to train.
    """
    fcfg = _fold_config(cfg, epochs)
    tft = fcfg["tft"]
    engineer = SpreadFeatureEngineer(fcfg)
    panel = engineer.engineer_pair(close_a, close_b, "fold", "fold", hedge_ratio=beta)
    if panel.empty:
        return pd.DataFrame()

    # Train only on rows whose date is in the train window (no test leakage).
    train_dates = set(close_a.index[:train_len])
    train_panel = panel[panel["datetime"].isin(train_dates)].copy()
    min_rows = tft["max_encoder_length"] + tft["max_prediction_length"] + 10
    if len(train_panel) < min_rows:
        logger.warning("Fold train window too short for TFT (%d rows)", len(train_panel))
        return pd.DataFrame()

    builder = TFTDatasetBuilder(fcfg)
    training, validation = builder.make_datasets(train_panel)
    predictor = TFTPredictor(fcfg)
    predictor.train(training, validation)

    # Rolling 1-step forecasts over the full fold; normalizers were fit on train.
    fc = predictor.predict_per_step(panel)
    fc = fc.merge(panel[["time_idx", "datetime"]], on="time_idx", how="left")
    return fc.set_index("datetime")[["prediction", "uncertainty"]]
