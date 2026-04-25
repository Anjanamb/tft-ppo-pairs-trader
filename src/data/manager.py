"""
Data manager — orchestrates all data sources, stores to DuckDB.
This is the single entry point for all data operations.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from src.data.yfinance_source import YFinanceSource
from src.data.crypto_source import CryptoSource
from src.utils.config import load_config

logger = logging.getLogger(__name__)


class DataManager:
    """Central data manager — fetches, stores, and serves market data."""

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        self.db_path = Path(self.cfg["data"]["db_path"])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._sources = {
            "yfinance": YFinanceSource(),
            "binance": CryptoSource("binance"),
        }

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def init_db(self):
        """Create tables if they don't exist."""
        con = self._connect()
        con.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                ticker VARCHAR,
                asset_class VARCHAR,
                datetime TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (ticker, datetime)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS pairs (
                pair_id VARCHAR PRIMARY KEY,
                ticker_a VARCHAR,
                ticker_b VARCHAR,
                asset_class_a VARCHAR,
                asset_class_b VARCHAR,
                cointegration_pvalue DOUBLE,
                correlation DOUBLE,
                half_life DOUBLE,
                last_updated TIMESTAMP
            )
        """)
        con.close()
        logger.info("Database initialized")

    def refresh_all(self):
        """Fetch latest data for all configured assets."""
        cfg_sources = self.cfg["data"]["sources"]
        history_days = self.cfg["data"]["history_days"]
        end = datetime.now()

        for asset_class, source_cfg in cfg_sources.items():
            if asset_class == "macro":
                continue  # TODO: implement FRED source

            provider = source_cfg["provider"]
            tickers = source_cfg["tickers"]
            interval = source_cfg["interval"]

            source = self._sources.get(provider)
            if source is None:
                logger.warning(f"No source for provider: {provider}")
                continue

            # Check what we already have
            start = self._get_last_date(tickers) or (
                end - timedelta(days=history_days)
            )

            logger.info(
                f"Refreshing {asset_class} ({len(tickers)} tickers) "
                f"from {start.date()} via {provider}"
            )

            data = source.fetch(tickers, start, end, interval)

            for ticker, df in data.items():
                self._upsert(ticker, asset_class, df)

        logger.info("Data refresh complete")

    def _get_last_date(self, tickers: list[str]) -> datetime | None:
        """Get the most recent date across tickers."""
        con = self._connect()
        placeholders = ", ".join(["?"] * len(tickers))
        try:
            result = con.execute(
                f"SELECT MAX(datetime) FROM ohlcv WHERE ticker IN ({placeholders})",
                tickers,
            ).fetchone()
            return result[0] if result and result[0] else None
        except Exception:
            return None
        finally:
            con.close()

    def _upsert(self, ticker: str, asset_class: str, df: pd.DataFrame):
        """Insert or update OHLCV data."""
        if df.empty:
            return

        con = self._connect()
        df = df.copy()
        df["ticker"] = ticker
        df["asset_class"] = asset_class
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "datetime"})

        # Use INSERT OR REPLACE via staging
        con.execute("CREATE TEMP TABLE IF NOT EXISTS staging AS SELECT * FROM ohlcv LIMIT 0")
        con.execute("DELETE FROM staging")
        con.register("df_view", df)
        con.execute("""
            INSERT INTO staging
            SELECT ticker, asset_class, datetime, open, high, low, close, volume
            FROM df_view
        """)
        con.execute("""
            INSERT OR REPLACE INTO ohlcv
            SELECT * FROM staging
        """)
        con.execute("DROP TABLE IF EXISTS staging")
        con.close()

        logger.info(f"  Stored {len(df)} rows for {ticker}")

    def get_prices(
        self,
        tickers: list[str],
        start: str | None = None,
        end: str | None = None,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Get a pivoted price DataFrame: DatetimeIndex x tickers.
        This is the primary interface for pair selection and modeling.
        """
        con = self._connect()
        where = "WHERE ticker IN (" + ", ".join(["?"] * len(tickers)) + ")"
        params = list(tickers)

        if start:
            where += " AND datetime >= ?"
            params.append(start)
        if end:
            where += " AND datetime <= ?"
            params.append(end)

        df = con.execute(
            f"SELECT datetime, ticker, {column} FROM ohlcv {where} ORDER BY datetime",
            params,
        ).fetchdf()
        con.close()

        if df.empty:
            return pd.DataFrame()

        return df.pivot(index="datetime", columns="ticker", values=column)

    def get_ohlcv(self, ticker: str, start: str | None = None) -> pd.DataFrame:
        """Get full OHLCV for a single ticker."""
        con = self._connect()
        query = "SELECT * FROM ohlcv WHERE ticker = ?"
        params = [ticker]
        if start:
            query += " AND datetime >= ?"
            params.append(start)
        query += " ORDER BY datetime"
        df = con.execute(query, params).fetchdf()
        con.close()
        return df.set_index("datetime") if not df.empty else df
