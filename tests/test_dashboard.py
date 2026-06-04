"""Tests for the dashboard data layer and a headless app smoke run."""

import pytest

from src.dashboard import data as D


# ----------------------------------------------------------------------
# Pure logic
# ----------------------------------------------------------------------
def test_classify_signal_thresholds():
    assert D.classify_signal(2.0) == "SHORT"    # rich spread
    assert D.classify_signal(-2.0) == "LONG"    # cheap spread
    assert D.classify_signal(0.1) == "FLAT"     # reverted
    assert D.classify_signal(0.7) == "HOLD"     # between bands
    assert D.classify_signal(float("nan")) == "NO DATA"


def test_spread_frame_columns_and_bands():
    pairs = D.load_pairs()
    if pairs.empty:
        pytest.skip("no pairs file available")
    row = pairs.iloc[0]
    sf = D.spread_frame(row["ticker_a"], row["ticker_b"], window=20)
    assert {"spread", "mean", "upper", "lower", "zscore"} <= set(sf.columns)
    valid = sf.dropna()
    # +2σ band sits above the rolling mean above -2σ band
    assert (valid["upper"] >= valid["mean"]).all()
    assert (valid["mean"] >= valid["lower"]).all()


def test_signals_table_has_valid_labels():
    pairs = D.load_pairs()
    if pairs.empty:
        pytest.skip("no pairs file available")
    sig = D.signals_table(pairs.head(3))
    assert set(sig.columns) >= {"pair_id", "zscore", "signal"}
    assert set(sig["signal"]) <= {"LONG", "SHORT", "FLAT", "HOLD", "NO DATA"}


# ----------------------------------------------------------------------
# Headless app — AppTest executes the script and surfaces exceptions
# ----------------------------------------------------------------------
def test_app_runs_without_exception():
    pytest.importorskip("streamlit")
    from pathlib import Path

    # The app reads the live DuckDB; skip on a fresh checkout / CI with no data.
    if not Path("data/market_data.duckdb").exists() or D.latest_pairs_file() is None:
        pytest.skip("no market data available")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("src/dashboard/app.py", default_timeout=60)
    at.run()
    assert not at.exception

    # exercise the fast pages (skip Model Performance: loads the TFT)
    for page in ["Pair Monitor", "Signals", "Backtest Results"]:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"page '{page}' raised"
