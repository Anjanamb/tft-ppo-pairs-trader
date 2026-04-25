"""
Base data source interface.
Adding a new data provider = implement this interface.

References:
    - yfinance for ETFs/stocks/commodities
    - ccxt for crypto exchanges
    - fredapi for macro indicators
"""

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class BaseDataSource(ABC):
    """Interface that all data providers must implement."""

    @abstractmethod
    def fetch(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for given tickers.

        Returns:
            Dict mapping ticker -> DataFrame with columns:
            [open, high, low, close, volume] and DatetimeIndex.
        """
        ...

    @abstractmethod
    def fetch_latest(
        self,
        tickers: list[str],
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Fetch the most recent data (for live/paper trading)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        ...
