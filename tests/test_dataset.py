"""Tests for the TFT dataset builder.

The TimeSeriesDataSet construction is skipped when pytorch-forecasting is not
installed, so the suite stays runnable in a lightweight environment.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.dataset import TFTDatasetBuilder
from src.models.features import SpreadFeatureEngineer


def _make_panel(n_pairs: int = 2, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2021-01-01", periods=n)
    eng = SpreadFeatureEngineer()

    panels = []
    for k in range(n_pairs):
        common = np.cumsum(rng.normal(0, 1, n)) + 100
        close_a = pd.Series(common + rng.normal(0, 0.5, n), index=index)
        close_b = pd.Series(common * 0.5 + 50 + rng.normal(0, 0.3, n), index=index)
        panels.append(
            eng.engineer_pair(close_a, close_b, f"P{k}A__P{k}B", "etfs-etfs")
        )
    return pd.concat(panels, ignore_index=True)


def test_make_datasets_builds_train_val():
    pytest.importorskip("pytorch_forecasting")

    from pytorch_forecasting import TimeSeriesDataSet

    panel = _make_panel()
    builder = TFTDatasetBuilder()
    training, validation = builder.make_datasets(panel)

    assert isinstance(training, TimeSeriesDataSet)
    assert isinstance(validation, TimeSeriesDataSet)
    assert len(training) > 0
    assert len(validation) > 0


def test_panel_has_all_pairs():
    panel = _make_panel(n_pairs=3)
    assert panel["pair_id"].nunique() == 3
    assert {"P0A__P0B", "P1A__P1B", "P2A__P2B"} == set(panel["pair_id"].unique())
