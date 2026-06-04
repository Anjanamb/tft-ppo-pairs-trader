"""Tests for the TFT spread predictor.

Skipped when pytorch-forecasting is unavailable. Uses a tiny network and a
single epoch on synthetic data so the round-trip stays fast and CPU-only.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from src.models.features import SpreadFeatureEngineer
from src.utils.config import load_config


def _tiny_config() -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["tft"].update(
        max_epochs=1,
        accelerator="cpu",
        hidden_size=8,
        hidden_continuous_size=4,
        attention_head_size=1,
        batch_size=32,
        max_encoder_length=20,
        max_prediction_length=3,
        early_stopping_patience=5,
    )
    return cfg


def _panel(n_pairs: int = 2, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2021-01-01", periods=n)
    eng = SpreadFeatureEngineer()
    panels = []
    for k in range(n_pairs):
        common = np.cumsum(rng.normal(0, 1, n)) + 100
        a = pd.Series(common + rng.normal(0, 0.5, n), index=index)
        b = pd.Series(common * 0.5 + 50 + rng.normal(0, 0.3, n), index=index)
        panels.append(eng.engineer_pair(a, b, f"P{k}A__P{k}B", "etfs-etfs"))
    return pd.concat(panels, ignore_index=True)


@pytest.fixture(scope="module")
def trained():
    pytest.importorskip("pytorch_forecasting")
    from src.models.dataset import TFTDatasetBuilder
    from src.models.tft_predictor import TFTPredictor

    cfg = _tiny_config()
    builder = TFTDatasetBuilder(cfg)
    training, validation = builder.make_datasets(_panel())

    predictor = TFTPredictor(cfg)
    metrics = predictor.train(training, validation)
    return predictor, validation, metrics


def test_train_returns_metrics(trained):
    _, _, metrics = trained
    assert metrics["epochs_run"] >= 1
    assert metrics["best_val_loss"] is None or metrics["best_val_loss"] >= 0


def test_predict_contract(trained):
    predictor, validation, _ = trained
    out = predictor.predict(validation)

    q_cols = [f"q_{int(q * 100):02d}" for q in predictor.quantiles]
    assert list(out.columns) == ["pair_id", "time_idx", "horizon", "prediction", *q_cols]
    assert not out.isnull().any().any()
    # prediction is the median quantile by construction
    assert np.allclose(out["prediction"], out["q_50"])
    # one row per (sample, horizon); horizons are 1..max_prediction_length
    assert set(out["horizon"].unique()) == {1, 2, 3}
    # NOTE: we deliberately do NOT assert non-crossing quantiles here.
    # QuantileLoss carries no monotonicity constraint, so an undertrained
    # model can cross — that is a model-quality signal, not a code invariant.
    # (On a properly trained run crossing drops to ~0%.)


def test_save_load_round_trip(trained, tmp_path):
    from src.models.tft_predictor import TFTPredictor

    predictor, validation, _ = trained
    out = predictor.predict(validation)

    ckpt = tmp_path / "tft.ckpt"
    predictor.save(ckpt)
    assert ckpt.exists()

    reloaded = TFTPredictor(_tiny_config())
    reloaded.load(ckpt)
    out2 = reloaded.predict(validation)

    assert np.allclose(
        out["prediction"].to_numpy(), out2["prediction"].to_numpy(), atol=1e-4
    )


def test_predict_before_train_raises():
    pytest.importorskip("pytorch_forecasting")
    from src.models.tft_predictor import TFTPredictor

    with pytest.raises(RuntimeError):
        TFTPredictor(_tiny_config()).predict(pd.DataFrame())
