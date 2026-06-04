"""
Streamlit dashboard for the TFT + PPO pairs trading system.

Thin rendering layer over ``src.dashboard.data`` (all logic lives there). Five
pages: Overview, Pair Monitor, Signals, Backtest Results, Model Performance.

Run:
    streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dashboard import data as D
from src.utils.config import load_config
from src.utils.runtime import configure_quiet_runtime

configure_quiet_runtime()  # silence Lightning banners/warnings in the dashboard

st.set_page_config(page_title="TFT-PPO Pairs Trader", layout="wide")

_SIGNAL_COLORS = {
    "LONG": "#1a9850", "SHORT": "#d73027",
    "FLAT": "#999999", "HOLD": "#fdae61", "NO DATA": "#cccccc",
}


# --- cached wrappers (logic stays in data.py) ---------------------------
@st.cache_data(show_spinner=False)
def cached_pairs():
    return D.load_pairs()


@st.cache_data(show_spinner=False)
def cached_coverage():
    return D.coverage_summary()


@st.cache_data(show_spinner=False)
def cached_spread(a, b, window):
    return D.spread_frame(a, b, window)


@st.cache_data(show_spinner=False)
def cached_signals(window):
    return D.signals_table(cached_pairs(), window)


@st.cache_data(show_spinner=True)
def cached_backtest(a, b):
    return D.zscore_backtest(a, b)


_INTRO = """
**Start with the basics — what is a "pair"?** A *pair* is **two assets that tend to
move together** because they're economically linked — for example Coca-Cola &
Pepsi, two big banks (JPMorgan & Goldman), or a gold ETF & gold futures. When two
assets normally track each other, their prices stay in a fairly steady
relationship.

**What is pairs trading?** Sometimes that relationship temporarily stretches — one
asset runs ahead of the other. Pairs trading bets that the gap will **snap back to
normal**: you go long the cheaper one and short the pricier one, and profit when
they re-converge. Crucially, it bets on the *relationship between the two*, not on
the overall market going up or down (so it can make money in a flat or falling
market). This app finds such pairs and studies how to trade them.

**The jargon, in plain English:**
- **Spread** — the gap between the two assets' prices. The strategy watches for it
  to get unusually wide, betting it will narrow again.
- **Z-score** — how unusual today's spread is. Near 0 = normal; beyond ±2 = stretched.
- **Cointegration / half-life** — statistical checks that the two assets *genuinely*
  track each other, and how fast the gap typically closes (in days).
- **Signal** — the system's *simulated* suggestion for a pair: **LONG** / **SHORT** the
  spread, or **FLAT** (stay out). No real money or orders are involved.
- **TFT** forecasts the spread a few days ahead with an uncertainty band; the **PPO**
  agent is a reinforcement-learning trader that decides when to act.

**Honest disclaimer:** this is a learning/portfolio project. Under rigorous
out-of-sample testing the strategy does **not** beat simply holding the S&P 500.
Nothing here is investment advice.
"""


# --- pages ---------------------------------------------------------------
def page_overview():
    st.title("Pairs Trading System — Overview")
    with st.expander("ℹ️  New here? Read this first (plain-English intro)", expanded=False):
        st.markdown(_INTRO)

    pairs = cached_pairs()
    cov = cached_coverage()

    c1, c2, c3 = st.columns(3)
    c1.metric("Tracked pairs", len(pairs))
    c2.metric("OHLCV rows", f"{cov['total_rows']:,}")
    c3.metric("Latest data", cov["last_date"])

    st.subheader("Asset coverage")
    if not cov["by_class"].empty:
        st.dataframe(cov["by_class"], width="stretch", hide_index=True)

    st.subheader("Top cointegrated pairs")
    if pairs.empty:
        st.info("No pairs yet — run scripts/find_pairs.py.")
    else:
        cols = ["pair_id", "correlation", "coint_pvalue", "half_life", "quality_score"]
        st.dataframe(pairs[cols].head(10), width="stretch", hide_index=True)


def page_pair_monitor():
    st.title("Pair Monitor")
    pairs = cached_pairs()
    if pairs.empty:
        st.info("No pairs available.")
        return

    pair_id = st.selectbox("Pair", pairs["pair_id"])
    window = st.slider("Z-score window (days)", 10, 60, 20, 5)
    row = pairs[pairs["pair_id"] == pair_id].iloc[0]
    sf = cached_spread(row["ticker_a"], row["ticker_b"], window)
    if sf.empty:
        st.warning("No price data for this pair.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sf.index, y=sf["spread"], name="spread"))
    fig.add_trace(go.Scatter(x=sf.index, y=sf["mean"], name="mean",
                             line=dict(dash="dot", color="gray")))
    fig.add_trace(go.Scatter(x=sf.index, y=sf["upper"], name="+2σ",
                             line=dict(dash="dash", color="red")))
    fig.add_trace(go.Scatter(x=sf.index, y=sf["lower"], name="-2σ",
                             line=dict(dash="dash", color="green")))
    fig.update_layout(height=420, margin=dict(t=30), legend_orientation="h")
    st.plotly_chart(fig, width="stretch")

    z = float(sf["zscore"].iloc[-1])
    signal = D.classify_signal(z)
    st.metric(f"Current z-score ({pair_id})", f"{z:.2f}", signal)


def page_signals():
    st.title("Live Signals (paper trading)")
    st.caption(
        "Recommendations from the z-score rule on the latest data — the only "
        "strategy that broke even under walk-forward. No orders are executed."
    )
    window = st.slider("Z-score window (days)", 10, 60, 20, 5)
    sig = cached_signals(window)
    if sig.empty:
        st.info("No pairs available.")
        return

    def color(val):
        return f"background-color: {_SIGNAL_COLORS.get(val, '#fff')}; color: white"

    st.dataframe(
        sig.style.map(color, subset=["signal"]),
        width="stretch", hide_index=True,
    )
    counts = sig["signal"].value_counts().to_dict()
    st.write({k: int(v) for k, v in counts.items()})


def page_backtest():
    st.title("Walk-Forward Backtest")
    st.caption(
        "Rolling train/test with the hedge ratio re-estimated per fold — every "
        "point is out-of-sample. Honest result: the strategy barely beats break "
        "even and trails SPY buy-and-hold."
    )
    pairs = cached_pairs()
    if pairs.empty:
        st.info("No pairs available.")
        return

    pair_id = st.selectbox("Pair", pairs["pair_id"], key="bt_pair")
    row = pairs[pairs["pair_id"] == pair_id].iloc[0]
    result = cached_backtest(row["ticker_a"], row["ticker_b"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["dates"], y=result["equity"],
                             name="z-score strategy (OOS PnL)"))
    fig.update_layout(height=360, margin=dict(t=30),
                      yaxis_title="cumulative spread PnL")
    st.plotly_chart(fig, width="stretch")

    m = result["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe", f"{m['sharpe_ratio']:.2f}")
    c2.metric("Max drawdown", f"{m['max_drawdown']:.2f}")
    c3.metric("Win rate", f"{m['win_rate']:.0%}")
    c4.metric("Folds", result["n_folds"])
    if result.get("benchmark"):
        st.caption(
            f"{result['benchmark_ticker']} buy-and-hold Sharpe over the same "
            f"dates: {result['benchmark']['sharpe_ratio']:.2f}"
        )
    with st.expander("Per-fold detail"):
        st.dataframe(result["per_fold"], width="stretch", hide_index=True)


@st.cache_resource(show_spinner=True)
def _load_predictor(ckpt_path):
    from src.models.tft_predictor import TFTPredictor

    predictor = TFTPredictor(load_config())
    predictor.load(ckpt_path)
    return predictor


def page_model():
    st.title("Model Performance — TFT forecasts")
    ckpts = sorted(Path("models").glob("tft_*.ckpt"))
    if not ckpts:
        st.info("No TFT checkpoint found — run scripts/train_tft.py.")
        return

    pairs = cached_pairs()
    pair_id = st.selectbox("Pair", pairs["pair_id"], key="tft_pair")

    from src.data.manager import DataManager
    from src.models.dataset import TFTDatasetBuilder

    cfg = load_config()
    panel = TFTDatasetBuilder(cfg).build_panel(
        pairs[pairs["pair_id"] == pair_id], DataManager(cfg)
    )
    predictor = _load_predictor(str(ckpts[-1]))
    forecasts = predictor.predict_per_step(panel)
    merged = panel.merge(forecasts, on=["pair_id", "time_idx"], how="inner")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=merged["spread"], name="actual spread"))
    fig.add_trace(go.Scatter(y=merged["prediction"], name="TFT median (q50)"))
    fig.add_trace(go.Scatter(
        y=merged["prediction"] + merged["uncertainty"] / 2,
        name="upper band", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(
        y=merged["prediction"] - merged["uncertainty"] / 2,
        name="uncertainty (q90-q10)", fill="tonexty",
        line=dict(width=0), fillcolor="rgba(31,119,180,0.2)"))
    fig.update_layout(height=420, margin=dict(t=30), legend_orientation="h")
    st.plotly_chart(fig, width="stretch")

    mae = float(np.abs(merged["spread"] - merged["prediction"]).mean())
    st.metric("Mean absolute forecast error (1-step)", f"{mae:.4f}")


PAGES = {
    "Overview": page_overview,
    "Pair Monitor": page_pair_monitor,
    "Signals": page_signals,
    "Backtest Results": page_backtest,
    "Model Performance": page_model,
}


def _data_controls():
    """Sidebar buttons to refresh data / rescan pairs without the CLI."""
    st.sidebar.divider()
    st.sidebar.subheader("Data controls")

    if st.sidebar.button("🔄 Refresh market data", width="stretch"):
        with st.spinner("Fetching latest bars from Binance / yfinance…"):
            last = D.refresh_market_data()
        st.cache_data.clear()  # invalidate cached prices/signals/backtests
        st.sidebar.success(f"Data updated through {last}")
        st.rerun()

    if st.sidebar.button("🔁 Rescan pairs", width="stretch"):
        with st.spinner("Scanning for cointegrated pairs…"):
            n = D.rescan_pairs()
        st.cache_data.clear()
        st.sidebar.success(f"Found {n} pairs")
        st.rerun()

    st.sidebar.caption("Data is a snapshot from the last refresh — not a live feed.")


def main():
    st.sidebar.title("TFT-PPO Pairs Trader")
    choice = st.sidebar.radio("Page", list(PAGES))
    st.sidebar.caption("Research dashboard — not investment advice.")
    _data_controls()
    PAGES[choice]()


# Streamlit executes this module top-to-bottom on every run.
main()
