"""
Base predictor interface.
Swap or ensemble models by implementing this interface.

Current: TFT (pytorch-forecasting)
Future: LSTM, N-BEATS, or any model that predicts spread dynamics.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BasePredictor(ABC):
    """Interface for spread prediction models."""

    @abstractmethod
    def train(self, train_data: Any, val_data: Any) -> dict:
        """
        Train the model.

        Args:
            train_data: Model-ready training data. The concrete type is left to
                the implementation — the TFT consumes prepared
                ``TimeSeriesDataSet`` objects, a simpler model might take a
                DataFrame.
            val_data: Validation data in the same form as ``train_data``.

        Returns:
            Dict of training metrics.
        """
        ...

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions with uncertainty quantiles.

        Returns:
            DataFrame with columns: [prediction, q_low, q_high, ...]
        """
        ...

    @abstractmethod
    def save(self, path: Path):
        """Save model artifacts."""
        ...

    @abstractmethod
    def load(self, path: Path):
        """Load model artifacts."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging and tracking."""
        ...
