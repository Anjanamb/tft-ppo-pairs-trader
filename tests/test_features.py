"""Tests for the spread feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.models.features import (
    KNOWN_REALS,
    STATIC_CATEGORICALS,
    UNKNOWN_REALS,
    SpreadFeatureEngineer,
)


@pytest.fixture
def pair_series():
    """Two correlated daily price series with a mean-reverting spread."""
    rng = np.random.default_rng(42)
    n = 300
    index = pd.bdate_range("2022-01-01", periods=n)

    common = np.cumsum(rng.normal(0, 1, n)) + 100
    noise = rng.normal(0, 0.5, n)
    close_a = pd.Series(common + noise, index=index)
    close_b = pd.Series(common * 0.5 + 50 + rng.normal(0, 0.3, n), index=index)
    volume_a = pd.Series(rng.uniform(1e6, 5e6, n), index=index)
    volume_b = pd.Series(rng.uniform(1e6, 5e6, n), index=index)
    return close_a, close_b, volume_a, volume_b


def test_engineer_pair_columns(pair_series):
    close_a, close_b, volume_a, volume_b = pair_series
    eng = SpreadFeatureEngineer()

    df = eng.engineer_pair(
        close_a, close_b, "A__B", "etfs-etfs", volume_a, volume_b
    )

    expected = {"datetime", "time_idx", *UNKNOWN_REALS, *KNOWN_REALS, *STATIC_CATEGORICALS}
    assert expected.issubset(df.columns)


def test_engineer_pair_no_nans_and_contiguous_index(pair_series):
    close_a, close_b, volume_a, volume_b = pair_series
    eng = SpreadFeatureEngineer()

    df = eng.engineer_pair(
        close_a, close_b, "A__B", "etfs-etfs", volume_a, volume_b
    )

    assert not df.empty
    assert not df[UNKNOWN_REALS].isnull().any().any()
    # time_idx must be a gap-free 0..n-1 range for pytorch-forecasting.
    assert df["time_idx"].tolist() == list(range(len(df)))


def test_volume_ratio_defaults_to_one(pair_series):
    close_a, close_b, _, _ = pair_series
    eng = SpreadFeatureEngineer()

    df = eng.engineer_pair(close_a, close_b, "A__B", "etfs-etfs")

    assert (df["volume_ratio"] == 1.0).all()


def test_too_few_observations_returns_empty():
    eng = SpreadFeatureEngineer()
    index = pd.bdate_range("2022-01-01", periods=10)
    short = pd.Series(np.arange(10, dtype=float), index=index)

    df = eng.engineer_pair(short, short + 1, "A__B", "etfs-etfs")

    assert df.empty
