"""
Pair selection engine.
Implements cointegration-based pair discovery across asset classes.

References:
    - Engle & Granger (1987) cointegration test
    - Han et al. (2023) "Select and Trade" — unified pair selection
"""

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller

from src.utils.config import load_config

logger = logging.getLogger(__name__)


class PairSelector:
    """Discovers and ranks tradeable pairs across asset classes."""

    def __init__(self, config: dict | None = None):
        self.cfg = (config or load_config())["pair_selection"]

    def find_pairs(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Scan all ticker combinations for cointegrated pairs.

        Args:
            prices: DataFrame with DatetimeIndex, columns = tickers, values = close prices.

        Returns:
            DataFrame of valid pairs sorted by quality score.
        """
        tickers = prices.columns.tolist()
        pairs_data = []

        total = len(list(combinations(tickers, 2)))
        logger.info(f"Scanning {total} pair combinations across {len(tickers)} assets")

        for i, (a, b) in enumerate(combinations(tickers, 2)):
            if i % 100 == 0:
                logger.info(f"  Progress: {i}/{total}")

            pa, pb = prices[a].dropna(), prices[b].dropna()
            common = pa.index.intersection(pb.index)
            if len(common) < self.cfg["lookback_window"]:
                continue

            pa, pb = pa.loc[common], pb.loc[common]
            result = self._evaluate_pair(a, b, pa, pb)
            if result is not None:
                pairs_data.append(result)

        if not pairs_data:
            logger.warning("No valid pairs found!")
            return pd.DataFrame()

        df = pd.DataFrame(pairs_data)
        df["quality_score"] = self._compute_quality_score(df)
        df = df.sort_values("quality_score", ascending=False).head(
            self.cfg["max_pairs"]
        )

        logger.info(f"Found {len(df)} valid pairs")
        return df.reset_index(drop=True)

    def _evaluate_pair(
        self, ticker_a: str, ticker_b: str, pa: pd.Series, pb: pd.Series
    ) -> dict | None:
        """Run cointegration and correlation checks on a candidate pair."""
        # Correlation check
        corr = pa.corr(pb)
        if abs(corr) < self.cfg["min_correlation"]:
            return None

        # Engle-Granger cointegration test
        try:
            score, pvalue, _ = coint(pa, pb)
        except Exception:
            return None

        if pvalue > 0.05:
            return None

        # Compute spread and half-life
        spread = self._compute_spread(pa, pb)
        half_life = self._compute_half_life(spread)

        if half_life is None:
            return None
        if half_life < self.cfg["min_half_life"] or half_life > self.cfg["max_half_life"]:
            return None

        # Spread stationarity check
        adf_stat, adf_pvalue, *_ = adfuller(spread.dropna(), maxlag=20)

        return {
            "pair_id": f"{ticker_a}__{ticker_b}",
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "correlation": round(corr, 4),
            "coint_pvalue": round(pvalue, 6),
            "coint_score": round(score, 4),
            "half_life": round(half_life, 2),
            "adf_pvalue": round(adf_pvalue, 6),
            "spread_std": round(spread.std(), 6),
            "n_observations": len(pa),
        }

    @staticmethod
    def _compute_spread(pa: pd.Series, pb: pd.Series) -> pd.Series:
        """Compute the log-price spread (hedge ratio via OLS)."""
        from numpy.polynomial.polynomial import polyfit

        log_a, log_b = np.log(pa), np.log(pb)
        coeffs = polyfit(log_b, log_a, 1)
        hedge_ratio = coeffs[1]
        return log_a - hedge_ratio * log_b

    @staticmethod
    def _compute_half_life(spread: pd.Series) -> float | None:
        """
        Ornstein-Uhlenbeck half-life of mean reversion.
        Shorter = faster reversion = better for trading.
        """
        spread = spread.dropna()
        lag = spread.shift(1).dropna()
        delta = spread.diff().dropna()

        common = lag.index.intersection(delta.index)
        lag, delta = lag.loc[common], delta.loc[common]

        if len(lag) < 20:
            return None

        from numpy.polynomial.polynomial import polyfit

        coeffs = polyfit(lag, delta, 1)
        theta = coeffs[1]

        if theta >= 0:
            return None  # not mean-reverting

        return -np.log(2) / theta

    @staticmethod
    def _compute_quality_score(df: pd.DataFrame) -> pd.Series:
        """
        Composite score: lower p-value + shorter half-life + higher correlation = better.
        Each component normalized to [0, 1] then averaged.
        """
        # Invert p-value (lower is better)
        pval_score = 1 - (df["coint_pvalue"] / df["coint_pvalue"].max()).clip(0, 1)

        # Invert half-life (shorter is better)
        hl_score = 1 - (
            (df["half_life"] - df["half_life"].min())
            / (df["half_life"].max() - df["half_life"].min() + 1e-8)
        )

        # Correlation (higher absolute value is better)
        corr_score = df["correlation"].abs()

        return (pval_score + hl_score + corr_score) / 3
