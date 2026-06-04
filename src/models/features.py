"""
Feature engineering for spread prediction.

Transforms raw close/volume series of a cointegrated pair into the covariate
panel the TFT consumes. The output is a long-format DataFrame (one row per
timestep) that ``src/models/dataset.py`` concatenates across pairs and wraps in
a ``pytorch_forecasting.TimeSeriesDataSet``.

References:
    - Lim et al. (2021) "Temporal Fusion Transformers" arXiv:1912.09363
    - Han et al. (2023) "Select and Trade" arXiv:2301.10724
"""

import logging

import numpy as np
import pandas as pd

from src.pairs.selector import PairSelector
from src.utils.config import load_config

logger = logging.getLogger(__name__)

# Time-varying covariates produced for every pair. Kept in sync with the
# ``tft.time_varying_*`` lists in config.yaml.
UNKNOWN_REALS = [
    "spread",
    "spread_zscore",
    "spread_ma_ratio",
    "volatility_ratio",
    "volume_ratio",
    "rsi_spread",
]
KNOWN_REALS = ["day_of_week", "month", "is_month_end"]
STATIC_CATEGORICALS = ["pair_id", "asset_class"]


class SpreadFeatureEngineer:
    """Builds TFT-ready feature panels from raw pair price/volume series."""

    def __init__(self, config: dict | None = None):
        self.cfg = (config or load_config())["features"]

    def engineer_pair(
        self,
        close_a: pd.Series,
        close_b: pd.Series,
        pair_id: str,
        asset_class: str,
        volume_a: pd.Series | None = None,
        volume_b: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Build the feature panel for a single pair.

        Args:
            close_a: Close prices for leg A (DatetimeIndex).
            close_b: Close prices for leg B (DatetimeIndex).
            pair_id: Group identifier, e.g. ``"BNB/USDT__XLF"``.
            asset_class: Static categorical label for the pair, e.g. ``"crypto-etfs"``.
            volume_a: Optional volume for leg A. Volume ratio is 1.0 if omitted.
            volume_b: Optional volume for leg B.

        Returns:
            Long-format DataFrame with a contiguous integer ``time_idx`` plus
            the spread target, covariates, calendar features, and statics. The
            warmup period (NaN rolling windows) is dropped.
        """
        idx = close_a.index.intersection(close_b.index)
        if len(idx) < self.cfg["zscore_window"] + self.cfg["ma_slow"]:
            logger.warning(
                "Pair %s has too few overlapping observations (%d); skipping",
                pair_id,
                len(idx),
            )
            return pd.DataFrame()

        close_a, close_b = close_a.loc[idx], close_b.loc[idx]

        # Reuse the selector's hedge-ratio spread so features and pair scoring
        # agree on what "the spread" is.
        spread = PairSelector._compute_spread(close_a, close_b)

        df = pd.DataFrame(index=idx)
        df["spread"] = spread
        df["spread_zscore"] = self._rolling_zscore(spread, self.cfg["zscore_window"])
        df["spread_ma_ratio"] = self._ma_ratio(
            spread, self.cfg["ma_fast"], self.cfg["ma_slow"]
        )
        df["volatility_ratio"] = self._volatility_ratio(
            close_a, close_b, self.cfg["volatility_window"]
        )
        df["volume_ratio"] = self._volume_ratio(volume_a, volume_b, idx)
        df["rsi_spread"] = self._rsi(spread, self.cfg["rsi_window"])

        # Calendar features (known into the future).
        df["day_of_week"] = df.index.dayofweek.astype("float32")
        df["month"] = df.index.month.astype("float32")
        df["is_month_end"] = df.index.is_month_end.astype("float32")

        # Statics.
        df["pair_id"] = pair_id
        df["asset_class"] = asset_class

        df = df.dropna()
        if df.empty:
            return df

        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "datetime"})
        df["time_idx"] = np.arange(len(df), dtype="int64")

        logger.info("Built %d feature rows for pair %s", len(df), pair_id)
        return df

    # ------------------------------------------------------------------
    # Feature primitives
    # ------------------------------------------------------------------
    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mean) / (std + 1e-8)

    @staticmethod
    def _ma_ratio(series: pd.Series, fast: int, slow: int) -> pd.Series:
        ma_fast = series.rolling(fast).mean()
        ma_slow = series.rolling(slow).mean()
        return ma_fast / (ma_slow.abs() + 1e-8)

    @staticmethod
    def _volatility_ratio(
        close_a: pd.Series, close_b: pd.Series, window: int
    ) -> pd.Series:
        """Rolling std of leg-A log returns over that of leg B."""
        ret_a = np.log(close_a).diff()
        ret_b = np.log(close_b).diff()
        vol_a = ret_a.rolling(window).std()
        vol_b = ret_b.rolling(window).std()
        return vol_a / (vol_b + 1e-8)

    @staticmethod
    def _volume_ratio(
        volume_a: pd.Series | None, volume_b: pd.Series | None, idx: pd.Index
    ) -> pd.Series:
        if volume_a is None or volume_b is None:
            return pd.Series(1.0, index=idx)
        va = volume_a.reindex(idx)
        vb = volume_b.reindex(idx)
        return va / (vb + 1e-8)

    @staticmethod
    def _rsi(series: pd.Series, window: int) -> pd.Series:
        """Wilder-style RSI computed on the spread level."""
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        return 100.0 - 100.0 / (1.0 + rs)
