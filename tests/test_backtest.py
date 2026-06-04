"""Tests for backtest metrics and the walk-forward engine."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.agents.evaluation import regime_zscore_policy
from src.backtest import metrics as M
from src.backtest.engine import WalkForwardBacktester, _hedge_ratio
from src.backtest.strategies import regime_zscore_strategy, zscore_strategy
from src.utils.config import load_config


# ----------------------------------------------------------------------
# Metrics (pure functions, exact arithmetic)
# ----------------------------------------------------------------------
def test_max_drawdown():
    # equity [1, 2, -1]; peak [1, 2, 2]; dd peak-equity max = 3
    assert M.max_drawdown(np.array([1.0, 1.0, -3.0])) == pytest.approx(3.0)


def test_win_rate_ignores_flat_days():
    assert M.win_rate(np.array([1.0, -1.0, 0.0, 2.0])) == pytest.approx(2 / 3)


def test_profit_factor():
    # gross profit 3, gross loss 2
    assert M.profit_factor(np.array([1.0, -1.0, 2.0, -1.0])) == pytest.approx(1.5)


def test_sharpe_zero_when_no_variance():
    assert M.sharpe_ratio(np.array([0.5, 0.5, 0.5])) == 0.0


def test_sortino_only_penalizes_downside():
    r = np.array([1.0, 1.0, -1.0])
    assert M.sortino_ratio(r) > 0
    # all-positive returns -> no downside -> guard returns 0.0
    assert M.sortino_ratio(np.array([1.0, 2.0, 3.0])) == 0.0


def test_compute_all_keys():
    m = M.compute_all(np.array([0.1, -0.05, 0.2, 0.0]))
    assert {"sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
            "win_rate", "profit_factor", "annualized_return"} <= set(m)


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------
def test_hedge_ratio_recovers_known_slope():
    b = np.arange(2, 60, dtype=float)
    a = b**2  # log a = 2 log b -> slope 2
    assert _hedge_ratio(a, b) == pytest.approx(2.0, abs=1e-6)


def _cointegrated_closes(n=160, seed=0):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 0.01, n))
    a = 100 * np.exp(common + rng.normal(0, 0.005, n))
    b = 50 * np.exp(common + rng.normal(0, 0.005, n))
    idx = pd.bdate_range("2021-01-01", periods=n)
    return pd.Series(a, index=idx), pd.Series(b, index=idx)


def _small_config():
    cfg = copy.deepcopy(load_config())
    cfg["backtest"]["walk_forward"] = {
        "train_window": 40, "test_window": 10, "step": 10
    }
    return cfg


def test_walk_forward_only_scores_test_windows():
    cfg = _small_config()
    a, b = _cointegrated_closes()
    bt = WalkForwardBacktester(cfg, warmup=5)
    result = bt.run(a, b, zscore_strategy())

    assert result["n_folds"] >= 1
    # each fold contributes exactly (test_window - 1) graded steps
    assert len(result["returns"]) == result["n_folds"] * (10 - 1)
    assert len(result["dates"]) == len(result["returns"])
    assert not result["per_fold"].empty
    # per-fold betas should vary as the window rolls (genuine re-estimation)
    assert result["per_fold"]["beta"].nunique() > 1


def _obs(spread, z):
    o = np.zeros(9)
    o[0], o[1] = spread, z
    return o


def test_regime_gate_blocks_non_mean_reverting():
    pol = regime_zscore_policy(entry=1.0, window=30, max_half_life=60)
    # a monotonic trend is NOT mean-reverting; an extreme z would say "short"
    actions = [pol(_obs(float(i), 3.0)) for i in range(40)]
    assert actions[-1] == 0  # gated to flat despite the extreme z-score


def test_regime_gate_allows_mean_reverting():
    pol = regime_zscore_policy(entry=1.0, window=40, max_half_life=60)
    rng = np.random.default_rng(0)
    x, actions = 0.0, []
    for _ in range(60):
        x = 0.7 * x + rng.normal(0, 0.1)  # fast mean reversion (half-life ~2d)
        actions.append(pol(_obs(x, 2.0)))
    assert actions[-1] == 2  # regime is healthy -> the short signal goes through


def test_walk_forward_regime_strategy_runs():
    cfg = _small_config()
    a, b = _cointegrated_closes()
    bt = WalkForwardBacktester(cfg, warmup=5)
    result = bt.run(a, b, regime_zscore_strategy(window=15))
    assert result["n_folds"] >= 1
    assert len(result["returns"]) == result["n_folds"] * (10 - 1)


def test_engine_threads_forecaster_through():
    # A fake forecaster verifies the per-fold forecast plumbing + date alignment
    # without the cost of training a TFT.
    cfg = _small_config()
    a, b = _cointegrated_closes(n=120)

    def fake_forecaster(cfg_, ca, cb, beta, train_len):
        sp = np.log(ca.to_numpy()) - beta * np.log(cb.to_numpy())
        return pd.DataFrame(
            {"prediction": sp + 0.01, "uncertainty": 0.1}, index=ca.index
        )

    bt = WalkForwardBacktester(cfg, warmup=5)
    result = bt.run(a, b, zscore_strategy(), forecaster=fake_forecaster)
    assert result["n_folds"] >= 1
    assert len(result["returns"]) == result["n_folds"] * (10 - 1)


def test_forecast_fold_returns_dated_forecasts():
    pytest.importorskip("pytorch_forecasting")
    from src.backtest.tft_fold import forecast_fold

    a, b = _cointegrated_closes(n=200)
    train_len = 140
    beta = _hedge_ratio(a.to_numpy()[:train_len], b.to_numpy()[:train_len])
    fc = forecast_fold(load_config(), a, b, beta, train_len, epochs=1)

    assert not fc.empty
    assert {"prediction", "uncertainty"} <= set(fc.columns)
    # forecasts extend into the held-out test window (dates after train_len)
    assert fc.index.max() >= a.index[train_len]
