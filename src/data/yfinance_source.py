"""
Yahoo Finance data source for ETFs, stocks, and commodities.
"""

import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

from src.data.base import BaseDataSource

logger = logging.getLogger(__name__)


class YFinanceSource(BaseDataSource):
    """Fetches data from Yahoo Finance via yfinance."""

    @property
    def name(self) -> str:
        return "yfinance"

    def fetch(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        result = {}
        for ticker in tickers:
            try:
                logger.info(f"Fetching {ticker} from {start} to {end}")
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )
                if df.empty:
                    logger.warning(f"No data returned for {ticker}")
                    continue

                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df.columns = [c.lower() for c in df.columns]
                df.index.name = "datetime"
                result[ticker] = df[["open", "high", "low", "close", "volume"]]

            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")

        return result

    def fetch_latest(
        self,
        tickers: list[str],
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        result = {}
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                df = t.history(period="5d", interval=interval)
                if df.empty:
                    continue
                df.columns = [c.lower() for c in df.columns]
                df.index.name = "datetime"
                result[ticker] = df[["open", "high", "low", "close", "volume"]]
            except Exception as e:
                logger.error(f"Failed to fetch latest {ticker}: {e}")
        return result
