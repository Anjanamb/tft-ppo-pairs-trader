"""
Walk-forward backtester.

For each fold the engine:
  1. refits the hedge ratio on the TRAIN closes only, then applies it to both
     train and test closes — this removes the look-ahead that a full-sample
     hedge ratio would bake into the spread definition;
  2. fits the strategy on a train-window environment;
  3. rolls the fitted policy through the held-out test window, warmed up with the
     tail of the train window so no test day is wasted and no future is used.

Test-fold returns are stitched into one out-of-sample series and scored. This is
the honest counterpart to the single holdout used in Phase 5 — every reported
return comes from data the strategy never trained on.

Costs (commission + slippage) are charged inside the environment per position
change. Market impact is not modeled (documented limitation).
"""

import logging

import numpy as np
import pandas as pd

from src.agents.evaluation import compute_metrics, run_episode
from src.agents.trading_env import PairsTradingEnv
from src.backtest import metrics as M
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def _hedge_ratio(close_a: np.ndarray, close_b: np.ndarray) -> float:
    """OLS slope of log(a) on log(b), estimated on the training window only."""
    log_a, log_b = np.log(close_a), np.log(close_b)
    return float(np.polyfit(log_b, log_a, 1)[0])


class WalkForwardBacktester:
    """Rolling train/test backtest with per-fold hedge-ratio re-estimation."""

    def __init__(self, config: dict | None = None, warmup: int = 20):
        self.cfg = config or load_config()
        wf = self.cfg["backtest"]["walk_forward"]
        self.train_window = wf["train_window"]
        self.test_window = wf["test_window"]
        self.step = wf["step"]
        self.warmup = warmup
        self.scaling = self.cfg["ppo"]["reward_scaling"]

    def run(
        self, close_a: pd.Series, close_b: pd.Series, strategy, forecaster=None
    ) -> dict:
        """
        Args:
            close_a, close_b: aligned close-price series for the pair.
            strategy: ``fit(train_env, cfg) -> policy_fn``.
            forecaster: optional ``(cfg, ca, cb, beta, train_len) -> DataFrame``
                refit each fold to produce look-ahead-free spread forecasts
                (datetime-indexed); they feed the test env's observation. If
                ``None`` the agent sees a zero-edge naive forecast.

        Returns:
            Dict with the stitched OOS return series, dates, metrics, per-fold
            metrics, and fold count.
        """
        idx = close_a.index.intersection(close_b.index)
        ca = close_a.loc[idx]
        cb = close_b.loc[idx]
        a = ca.to_numpy()
        b = cb.to_numpy()
        dates = idx

        n = len(a)
        all_returns: list[np.ndarray] = []
        all_dates: list[pd.Timestamp] = []
        per_fold = []

        start = 0
        fold = 0
        while start + self.train_window + self.test_window <= n:
            tr_end = start + self.train_window
            te_end = tr_end + self.test_window

            beta = _hedge_ratio(a[start:tr_end], b[start:tr_end])
            train_spread = np.log(a[start:tr_end]) - beta * np.log(b[start:tr_end])
            # prepend warmup tail of train so test features are warm and causal
            test_lo = tr_end - self.warmup
            test_spread = np.log(a[test_lo:te_end]) - beta * np.log(b[test_lo:te_end])

            forecast, uncertainty = None, None
            if forecaster is not None:
                fc = forecaster(
                    self.cfg, ca.iloc[start:te_end], cb.iloc[start:te_end],
                    beta, self.train_window,
                )
                forecast, uncertainty = self._align_forecast(
                    fc, dates[test_lo:te_end], test_spread
                )

            train_env = PairsTradingEnv(
                train_spread, config=self.cfg, warmup=self.warmup
            )
            test_env = PairsTradingEnv(
                test_spread, forecast, uncertainty,
                config=self.cfg, warmup=self.warmup,
            )
            policy = strategy(train_env, self.cfg)
            ep = run_episode(test_env, policy)

            returns = np.asarray(ep["rewards"], dtype=float) / self.scaling
            fold_dates = dates[tr_end + 1 : tr_end + 1 + len(returns)]
            all_returns.append(returns)
            all_dates.extend(fold_dates)
            per_fold.append(
                {
                    "fold": fold,
                    "train_end": str(dates[tr_end - 1].date()),
                    "test_end": str(dates[te_end - 1].date()),
                    "beta": round(beta, 4),
                    **compute_metrics(ep["rewards"], self.scaling, ep["info"]),
                }
            )
            logger.info(
                "Fold %d | test->%s | beta=%.3f | OOS Sharpe=%.2f | trades=%d",
                fold, per_fold[-1]["test_end"], beta,
                per_fold[-1]["sharpe"], per_fold[-1]["n_trades"],
            )
            start += self.step
            fold += 1

        returns = np.concatenate(all_returns) if all_returns else np.array([])
        return {
            "returns": returns,
            "dates": pd.DatetimeIndex(all_dates),
            "metrics": M.compute_all(returns),
            "per_fold": pd.DataFrame(per_fold),
            "n_folds": fold,
        }

    @staticmethod
    def _align_forecast(fc: pd.DataFrame, test_dates, test_spread):
        """Map a datetime-indexed forecast onto the test window positions.

        Missing dates (early window / no forecast) fall back to a zero-edge
        naive forecast so the agent simply sees no signal there.
        """
        forecast = test_spread.copy()
        uncertainty = np.zeros(len(test_spread))
        if fc is None or fc.empty:
            return forecast, uncertainty

        pred = fc["prediction"].reindex(test_dates).to_numpy()
        unc = fc["uncertainty"].reindex(test_dates).to_numpy()
        have = ~np.isnan(pred)
        forecast[have] = pred[have]
        med = np.nanmedian(unc) if np.any(~np.isnan(unc)) else 0.0
        unc[np.isnan(unc)] = med
        return forecast, unc


def benchmark_metrics(
    close: pd.Series, dates: pd.DatetimeIndex
) -> dict | None:
    """Buy-and-hold daily-return metrics for a benchmark over the OOS dates."""
    if dates is None or len(dates) == 0:
        return None
    series = close.reindex(close.index.union(dates)).interpolate()
    window = series.reindex(dates)
    rets = np.log(window).diff().dropna().to_numpy()
    return M.compute_all(rets)
