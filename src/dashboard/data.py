"""
Dashboard data layer — pure logic, no Streamlit imports.

Keeping this module free of ``streamlit`` means every function here is unit
testable without a UI runtime; ``app.py`` is a thin caching/rendering shell on
top of it.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import WalkForwardBacktester, _hedge_ratio, benchmark_metrics
from src.backtest.strategies import zscore_strategy
from src.data.manager import DataManager
from src.pairs.selector import PairSelector
from src.utils.config import get_all_tickers_flat, load_config

logger = logging.getLogger(__name__)

# Signal thresholds (z-score units) — mirror agents.evaluation.zscore_policy.
ENTRY_Z = 1.0
EXIT_Z = 0.5


def latest_pairs_file() -> Path | None:
    files = sorted(Path("data/pairs").glob("pairs_*.csv"))
    return files[-1] if files else None


def load_pairs() -> pd.DataFrame:
    """Discovered pairs, ranked by quality score (empty frame if none yet)."""
    path = latest_pairs_file()
    return pd.read_csv(path) if path else pd.DataFrame()


def coverage_summary(cfg: dict | None = None) -> dict:
    """Row/ticker counts per asset class straight from DuckDB."""
    cfg = cfg or load_config()
    dm = DataManager(cfg)
    con = dm._connect()
    try:
        df = con.execute(
            "SELECT asset_class, COUNT(DISTINCT ticker) AS tickers, "
            "COUNT(*) AS rows, MAX(datetime)::DATE AS last_date "
            "FROM ohlcv GROUP BY asset_class ORDER BY asset_class"
        ).fetchdf()
    finally:
        con.close()
    return {
        "by_class": df,
        "total_rows": int(df["rows"].sum()) if not df.empty else 0,
        "last_date": str(df["last_date"].max()) if not df.empty else "—",
    }


def spread_frame(
    ticker_a: str, ticker_b: str, window: int = 20, cfg: dict | None = None
) -> pd.DataFrame:
    """
    Spread plus rolling mean / +-2 sigma bands / z-score for the monitor chart.

    The hedge ratio is fit on the full sample here — fine for a monitoring view,
    unlike the backtest which re-estimates it per fold.
    """
    cfg = cfg or load_config()
    prices = DataManager(cfg).get_prices([ticker_a, ticker_b]).dropna()
    if prices.empty or ticker_a not in prices or ticker_b not in prices:
        return pd.DataFrame()

    a, b = prices[ticker_a].to_numpy(), prices[ticker_b].to_numpy()
    beta = _hedge_ratio(a, b)
    spread = np.log(a) - beta * np.log(b)
    s = pd.Series(spread, index=prices.index)
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    return pd.DataFrame(
        {
            "spread": s,
            "mean": mean,
            "upper": mean + 2 * std,
            "lower": mean - 2 * std,
            "zscore": (s - mean) / (std + 1e-8),
        }
    )


def classify_signal(zscore: float, entry: float = ENTRY_Z, exit_: float = EXIT_Z) -> str:
    """Map a z-score to a paper-trading recommendation."""
    if np.isnan(zscore):
        return "NO DATA"
    if zscore > entry:
        return "SHORT"   # spread rich -> short the spread
    if zscore < -entry:
        return "LONG"    # spread cheap -> long the spread
    if abs(zscore) < exit_:
        return "FLAT"
    return "HOLD"


def signals_table(
    pairs_df: pd.DataFrame, window: int = 20, cfg: dict | None = None
) -> pd.DataFrame:
    """Current paper-trading signal for each tracked pair (latest z-score)."""
    cfg = cfg or load_config()
    rows = []
    for _, p in pairs_df.iterrows():
        sf = spread_frame(p["ticker_a"], p["ticker_b"], window, cfg)
        z = float(sf["zscore"].iloc[-1]) if not sf.empty else float("nan")
        rows.append(
            {
                "pair_id": p["pair_id"],
                "half_life": p.get("half_life"),
                "zscore": round(z, 2) if not np.isnan(z) else None,
                "signal": classify_signal(z),
            }
        )
    return pd.DataFrame(rows)


def zscore_backtest(
    ticker_a: str, ticker_b: str, cfg: dict | None = None
) -> dict:
    """Walk-forward backtest with the z-score rule + SPY benchmark."""
    cfg = cfg or load_config()
    dm = DataManager(cfg)
    prices = dm.get_prices([ticker_a, ticker_b]).dropna()
    bt = WalkForwardBacktester(cfg)
    result = bt.run(prices[ticker_a], prices[ticker_b], zscore_strategy())

    bench_ticker = cfg["backtest"]["benchmark"]
    bench_close = dm.get_prices([bench_ticker]).get(bench_ticker)
    result["benchmark"] = (
        benchmark_metrics(bench_close, result["dates"])
        if bench_close is not None else None
    )
    result["benchmark_ticker"] = bench_ticker
    result["equity"] = np.cumsum(result["returns"])
    return result


# ----------------------------------------------------------------------
# Actions the dashboard can trigger (same work as the cron scripts)
# ----------------------------------------------------------------------
def refresh_market_data(cfg: dict | None = None) -> str:
    """Fetch the latest bars for all configured tickers into DuckDB.

    Mirrors scripts/data_refresh.py. Returns the new latest data date.
    """
    cfg = cfg or load_config()
    dm = DataManager(cfg)
    dm.init_db()
    dm.refresh_all()
    return coverage_summary(cfg)["last_date"]


def rescan_pairs(cfg: dict | None = None) -> int:
    """Re-run cointegration pair discovery and save a dated CSV.

    Mirrors scripts/find_pairs.py. Returns the number of pairs found.
    """
    cfg = cfg or load_config()
    dm = DataManager(cfg)
    selector = PairSelector(cfg)

    prices = dm.get_prices(get_all_tickers_flat())
    if prices.empty:
        return 0
    min_obs = cfg["pair_selection"]["lookback_window"]
    valid = prices.columns[prices.count() >= min_obs]
    pairs_df = selector.find_pairs(prices[valid].dropna(how="all"))

    if not pairs_df.empty:
        out_dir = Path("data/pairs")
        out_dir.mkdir(parents=True, exist_ok=True)
        pairs_df.to_csv(out_dir / f"pairs_{datetime.now():%Y%m%d}.csv", index=False)
    return len(pairs_df)
