"""
Base predictor interface.
Swap or ensemble models by implementing this interface.

Current: TFT (pytorch-forecasting)
Future: LSTM, N-BEATS, or any model that predicts spread dynamics.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BasePredictor(ABC):
    """Interface for spread prediction models."""

    @abstractmethod
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> dict:
        """
        Train the model.

        Args:
            train_data: Training features + targets.
            val_data: Validation data.

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
