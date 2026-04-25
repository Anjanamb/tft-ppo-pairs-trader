"""
Crypto exchange data source via ccxt.
Supports Binance (default), extensible to other exchanges.
"""

import logging
from datetime import datetime

import pandas as pd

from src.data.base import BaseDataSource

logger = logging.getLogger(__name__)


class CryptoSource(BaseDataSource):
    """Fetches crypto OHLCV data via ccxt."""

    def __init__(self, exchange_id: str = "binance"):
        self._exchange_id = exchange_id
        self._exchange = None

    def _get_exchange(self):
        if self._exchange is None:
            import ccxt
            exchange_class = getattr(ccxt, self._exchange_id)
            self._exchange = exchange_class({"enableRateLimit": True})
        return self._exchange

    @property
    def name(self) -> str:
        return f"ccxt-{self._exchange_id}"

    def fetch(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        interval: str = "1h",
    ) -> dict[str, pd.DataFrame]:
        exchange = self._get_exchange()
        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        result = {}
        for symbol in tickers:
            try:
                logger.info(f"Fetching {symbol} from {self._exchange_id}")
                all_candles = []
                current_since = since_ms

                while current_since < end_ms:
                    candles = exchange.fetch_ohlcv(
                        symbol, interval, since=current_since, limit=1000
                    )
                    if not candles:
                        break
                    all_candles.extend(candles)
                    current_since = candles[-1][0] + 1

                if not all_candles:
                    logger.warning(f"No data for {symbol}")
                    continue

                df = pd.DataFrame(
                    all_candles,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.set_index("datetime").drop(columns=["timestamp"])
                df = df[df.index <= pd.Timestamp(end)]
                result[symbol] = df

            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")

        return result

    def fetch_latest(
        self,
        tickers: list[str],
        interval: str = "1h",
    ) -> dict[str, pd.DataFrame]:
        exchange = self._get_exchange()
        result = {}
        for symbol in tickers:
            try:
                candles = exchange.fetch_ohlcv(symbol, interval, limit=100)
                if not candles:
                    continue
                df = pd.DataFrame(
                    candles,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.set_index("datetime").drop(columns=["timestamp"])
                result[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch latest {symbol}: {e}")
        return result
