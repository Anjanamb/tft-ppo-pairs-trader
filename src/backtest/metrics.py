"""
Performance metrics for backtest return series.

All functions operate on an array of per-period (daily) returns and are pure, so
they are trivially testable. The strategy "returns" are additive spread PnL (not
compounding a price), so annualization is arithmetic (mean * periods).
"""

import numpy as np

_PERIODS = 252  # trading days / year


def sharpe_ratio(returns: np.ndarray, periods: int = _PERIODS) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods))


def sortino_ratio(returns: np.ndarray, periods: int = _PERIODS) -> float:
    r = np.asarray(returns, dtype=float)
    downside = np.minimum(r, 0.0)
    dd = np.sqrt(np.mean(downside**2))
    if r.size == 0 or dd == 0:
        return 0.0
    return float(r.mean() / dd * np.sqrt(periods))


def max_drawdown(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    equity = np.cumsum(r)
    peak = np.maximum.accumulate(equity)
    return float((peak - equity).max())


def annualized_return(returns: np.ndarray, periods: int = _PERIODS) -> float:
    r = np.asarray(returns, dtype=float)
    return float(r.mean() * periods) if r.size else 0.0


def calmar_ratio(returns: np.ndarray, periods: int = _PERIODS) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return annualized_return(returns, periods) / mdd


def win_rate(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    active = r[r != 0]
    if active.size == 0:
        return 0.0
    return float((active > 0).mean())


def profit_factor(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    gross_profit = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def compute_all(returns: np.ndarray) -> dict:
    """Full metric suite used by the backtest report (config `backtest.metrics`)."""
    r = np.asarray(returns, dtype=float)
    return {
        "total_return": float(r.sum()),
        "annualized_return": annualized_return(r),
        "sharpe_ratio": sharpe_ratio(r),
        "sortino_ratio": sortino_ratio(r),
        "max_drawdown": max_drawdown(r),
        "calmar_ratio": calmar_ratio(r),
        "win_rate": win_rate(r),
        "profit_factor": profit_factor(r),
        "n_periods": int(r.size),
    }
