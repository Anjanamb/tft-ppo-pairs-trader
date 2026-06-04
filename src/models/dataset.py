"""
TFT dataset builder.

Turns the per-pair feature panels from ``features.py`` into a single long
DataFrame across all tracked pairs, then wraps it in the
``pytorch_forecasting.TimeSeriesDataSet`` pair (train + validation) that the
TFT predictor trains on.

``pytorch_forecasting`` is imported lazily so that feature engineering and the
panel builder remain usable without the deep-learning stack installed.
"""

import logging

import pandas as pd

from src.data.manager import DataManager
from src.models.features import (
    KNOWN_REALS,
    STATIC_CATEGORICALS,
    UNKNOWN_REALS,
    SpreadFeatureEngineer,
)
from src.utils.config import get_asset_class, load_config

logger = logging.getLogger(__name__)


class TFTDatasetBuilder:
    """Assembles the multi-pair panel and TimeSeriesDataSets for the TFT."""

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        self.engineer = SpreadFeatureEngineer(self.cfg)

    def build_panel(
        self, pairs_df: pd.DataFrame, dm: DataManager | None = None
    ) -> pd.DataFrame:
        """
        Build a long-format panel covering every pair in ``pairs_df``.

        Args:
            pairs_df: Output of ``PairSelector.find_pairs`` — needs ``pair_id``,
                ``ticker_a``, ``ticker_b`` columns.
            dm: DataManager to pull prices from. Created if not supplied.

        Returns:
            Concatenated feature panel ready for ``make_datasets``. Empty if no
            pair yielded usable data.
        """
        dm = dm or DataManager(self.cfg)
        panels = []

        for _, row in pairs_df.iterrows():
            ticker_a, ticker_b = row["ticker_a"], row["ticker_b"]
            prices = dm.get_prices([ticker_a, ticker_b])
            volumes = dm.get_prices([ticker_a, ticker_b], column="volume")

            if prices.empty or ticker_a not in prices or ticker_b not in prices:
                logger.warning("Missing price data for pair %s; skipping", row["pair_id"])
                continue

            asset_class = f"{get_asset_class(ticker_a)}-{get_asset_class(ticker_b)}"
            panel = self.engineer.engineer_pair(
                close_a=prices[ticker_a].dropna(),
                close_b=prices[ticker_b].dropna(),
                pair_id=row["pair_id"],
                asset_class=asset_class,
                volume_a=volumes.get(ticker_a),
                volume_b=volumes.get(ticker_b),
            )
            if not panel.empty:
                panels.append(panel)

        if not panels:
            logger.warning("No pairs produced feature panels")
            return pd.DataFrame()

        combined = pd.concat(panels, ignore_index=True)
        logger.info(
            "Built panel: %d rows across %d pairs",
            len(combined),
            combined["pair_id"].nunique(),
        )
        return combined

    def make_datasets(self, panel: pd.DataFrame):
        """
        Build the training and validation ``TimeSeriesDataSet`` from a panel.

        The last ``features.val_fraction`` of each pair's history (by time index)
        is reserved for validation; the split respects the encoder lookback so
        validation windows have enough history.

        Returns:
            Tuple ``(training, validation)`` of TimeSeriesDataSet objects.
        """
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer

        tft_cfg = self.cfg["tft"]
        encoder_len = tft_cfg["max_encoder_length"]
        pred_len = tft_cfg["max_prediction_length"]
        val_fraction = self.cfg["features"]["val_fraction"]

        # Cutoff measured on the shortest pair so every group keeps a train span.
        max_idx = int(panel["time_idx"].max())
        cutoff = int(max_idx * (1 - val_fraction))

        training = TimeSeriesDataSet(
            panel[panel["time_idx"] <= cutoff],
            time_idx="time_idx",
            target="spread",
            group_ids=["pair_id"],
            max_encoder_length=encoder_len,
            max_prediction_length=pred_len,
            static_categoricals=STATIC_CATEGORICALS,
            time_varying_known_reals=KNOWN_REALS,
            time_varying_unknown_reals=UNKNOWN_REALS,
            target_normalizer=GroupNormalizer(groups=["pair_id"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

        validation = TimeSeriesDataSet.from_dataset(
            training, panel, predict=False, stop_randomization=True
        )

        logger.info(
            "Datasets ready — train cutoff time_idx=%d (max=%d)", cutoff, max_idx
        )
        return training, validation
