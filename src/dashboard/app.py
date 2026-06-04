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


# --- pages ---------------------------------------------------------------
def page_overview():
    st.title("Pairs Trading System — Overview")
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


def main():
    st.sidebar.title("TFT-PPO Pairs Trader")
    choice = st.sidebar.radio("Page", list(PAGES))
    st.sidebar.caption("Research dashboard — not investment advice.")
    PAGES[choice]()


# Streamlit executes this module top-to-bottom on every run.
main()
