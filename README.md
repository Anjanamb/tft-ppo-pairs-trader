# tft-ppo-pairs-trader

Multi-asset pairs trading with TFT spread prediction and PPO execution.

[![CI](https://github.com/Anjanamb/tft-ppo-pairs-trader/actions/workflows/ci.yml/badge.svg)](https://github.com/Anjanamb/tft-ppo-pairs-trader/actions/workflows/ci.yml)
[![Status](https://img.shields.io/badge/status-active%20development-yellow?style=flat-square)](#roadmap)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=flat-square&logo=duckdb&logoColor=black)](https://duckdb.org/)

The idea: use a [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363) to forecast the spread between cointegrated asset pairs, then let a PPO agent decide when to enter/exit trades based on those forecasts + uncertainty estimates. Runs across crypto (Binance), US equities, ETFs, and commodities.

> **Status:** End-to-end pipeline is built and CI-tested — data → pairs → TFT forecaster → PPO agent → Optuna tuning → walk-forward backtest → Streamlit dashboard, containerized with a GitHub Actions CI. The headline finding is deliberately unvarnished: **under a proper walk-forward the strategy does not beat a SPY buy-and-hold** (see [Results](#results)). Remaining: alerts/paper-trading and the modeling work to actually earn an edge. See [Roadmap](#roadmap).

## Why?

Most pairs trading implementations use simple z-score thresholds. That works, but leaves alpha on the table — the entry/exit thresholds are static, there's no uncertainty awareness, and the strategy can't adapt to regime changes.

Here, the TFT provides multi-horizon spread forecasts with quantile uncertainty (so the agent knows when it's confident vs guessing), and the PPO agent learns a policy that adapts to market conditions rather than following fixed rules.

The multi-asset angle is the other differentiator. Running cointegration scans across crypto, equities, ETFs, and commodities surfaces pairs that single-asset-class strategies miss entirely (e.g., BNB/USDT ↔ XLF turned up as the highest-scoring pair in initial scans — a crypto exchange token cointegrated with the US financials ETF).

## Papers behind this

- Lim et al. (2021) — [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)
- Han et al. (2023) — [Select and Trade: Towards Unified Pair Trading with Hierarchical Reinforcement Learning](https://arxiv.org/abs/2301.10724)
- Peik et al. (2025) — [Adaptive Temporal Fusion Transformer for Cryptocurrency](https://arxiv.org/abs/2509.10542), which flags TFT + RL integration as future work

## Architecture

```
config.yaml ──┐  (tickers, params, schedules)
              ▼
    ┌──────────────────┐
    │  Data Sources    │  Binance · yfinance · FRED
    │  → DuckDB store  │
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │  Pair Selection  │  cointegration · correlation · half-life
    └────────┬─────────┘
             ▼
    ┌──────────────────┐         ┌──────────────────┐
    │   TFT Predictor  │────────▶│    PPO Agent     │
    │  spread + quantile        │  entry · exit ·   │
    │     uncertainty           │  position sizing  │
    └──────────────────┘         └────────┬─────────┘
                                          ▼
                  Backtester · Paper trader · Signals · Dashboard
```

Adding a new ticker = one line in `config.yaml`. Swapping the RL algorithm = change `algorithm: "PPO"` to `"SAC"` or `"A2C"` (Stable-Baselines3 handles the rest). New prediction model = implement the `BasePredictor` interface.

## Setup

Needs Python 3.11 (pytorch-forecasting has issues on 3.13). If you're on conda:

```bash
conda create -n pairs-trader python=3.11 -y
conda activate pairs-trader

# PyTorch first. Only `torch` is needed (no torchvision/torchaudio).
# CPU build runs the whole project and is the simplest path:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For an NVIDIA GPU, pick the CUDA build matching your driver from
# https://pytorch.org/get-started/locally/  (e.g. cu124). The project was
# built and tested against torch 2.5–2.6.

# Everything else (pytorch-forecasting, lightning, SB3, gymnasium, ...):
pip install -r requirements.txt
```

> If a `--index-url` install fails with `No matching distribution found`, it's
> almost always a transient network/proxy hiccup — just retry the command.

## Usage

```bash
# 1. Pull 5 years of data for all configured tickers
python scripts/data_refresh.py

# 2. Scan for cointegrated pairs
python scripts/find_pairs.py

# 3. Train the TFT spread forecaster (quantile outputs + variable importance)
python scripts/train_tft.py --top 5 --interpret

# 4. Train the PPO agent on the forecasts (auto-uses the latest TFT checkpoint)
python scripts/train_ppo.py

# 5. Tune PPO against out-of-sample Sharpe (Optuna + MLflow + SQLite)
python -m src.tuning.optimizer --target ppo --n-trials 50

# 6. Walk-forward backtest vs a SPY benchmark (hedge ratio refit per fold)
python scripts/run_backtest.py --strategy both        # rule + PPO baselines
python scripts/run_backtest.py --strategy ppo --tft   # PPO on a per-fold TFT

# 7. Launch the dashboard
streamlit run src/dashboard/app.py
```

## Tests, CI & Docker

```bash
pytest                       # 35 tests; data-dependent ones skip without a DB
ruff check src/ scripts/ tests/

docker compose up dashboard           # serve the dashboard on :8501
docker compose run --rm data-refresh  # one-shot data pull
```

GitHub Actions runs ruff + the full test suite on every push (CPU-only PyTorch;
synthetic model/agent/backtest tests train tiny models for real).

## What's in the box

```
configs/config.yaml        <- all tickers, model params, schedules
src/data/                  <- data sources (yfinance, ccxt) + DuckDB manager
src/pairs/selector.py      <- cointegration tests, half-life, pair ranking
src/models/                <- feature engineering, TFT dataset builder, TFT predictor
src/agents/                <- Gymnasium env, SB3 agent wrapper, evaluation + baselines
src/tuning/                <- Optuna search spaces + optimizer (OOS-Sharpe objective)
src/backtest/              <- walk-forward engine, metrics, strategies
src/dashboard/             <- Streamlit app (logic in data.py, UI in app.py)
scripts/                   <- entry points (data refresh, pair scan, train, backtest)
tests/                     <- pytest suite (synthetic data; runs in CI)
```

## Results

**Pair discovery** — scan across 22 assets surfaced 12 cointegrated pairs, including cross-asset (crypto ↔ equity) and negatively correlated ones:

```
BNB/USDT ↔ XLF     corr=0.888  coint_p=0.010  half_life=21.0d  score=0.892
GC=F     ↔ GS      corr=0.967  coint_p=0.006  half_life=33.4d  score=0.839
GLD      ↔ GS      corr=0.966  coint_p=0.006  half_life=34.5d  score=0.831
JPM      ↔ SPY     corr=0.984  coint_p=0.021  half_life=28.5d  score=0.784
AAPL     ↔ NVDA    corr=0.933  coint_p=0.008  half_life=39.8d  score=0.756
```

**TFT forecaster** — trains across all pairs at once with per-pair target normalization and 7-quantile output. Variable importance is sensible: the spread level (35%) and its z-score (16%) dominate, exactly what cointegration theory predicts; engineered volume/volatility ratios add little.

**PPO agent vs baselines (single holdout, BNB/USDT ↔ XLF)** — PPO beat a z-score rule and random, the only policy positive out-of-sample (Sharpe 1.14 OOS vs 4.47 in-sample — the gap is the overfitting tax, shown rather than hidden).

**The honest verdict — walk-forward (12 folds, 744 OOS days, fully look-ahead-free):**

Every strategy is refit per fold, the hedge ratio is re-estimated per fold, and the TFT is retrained per fold on train-only data — so nothing below has seen its own evaluation window.

| Strategy                     | Ann. return | Sharpe | Max DD | Profit factor |
|------------------------------|------------:|-------:|-------:|--------------:|
| z-score rule                 |       +0.11 |   0.21 |   0.61 |          1.05 |
| z-score + regime filter      |       +0.08 |   0.18 |   0.62 |          1.04 |
| PPO + per-fold TFT forecasts |       −0.30 |  −0.56 |   1.23 |          0.90 |
| SPY buy-and-hold (benchmark) |       +0.20 |   1.32 |   0.21 |          1.28 |

Under rigorous evaluation the result is unambiguous: the z-score rule barely breaks even, a mean-reversion **regime filter does not help**, and the **PPO agent — even fed genuinely look-ahead-free TFT forecasts retrained every fold — loses money** and gets nowhere near just holding SPY. The flattering single-holdout result (Sharpe 1.14) did not survive, and successive attempts to rescue it (regime gating, per-fold TFT) confirmed there is no tradeable edge here.

That *is* the finding. The engineering value of this project is a complete, CI-tested pipeline that **evaluates honestly enough to kill its own false positives** — the opposite of a backtest that quietly leaks the future to manufacture a Sharpe.

## Data

Stored in DuckDB locally. Current dataset:

- **Crypto**: ~44k hourly candles per ticker (BTC, ETH, BNB, SOL, ADA)
- **Equities/ETFs/Commodities**: ~1,255 daily bars per ticker (SPY, QQQ, AAPL, MSFT, NVDA, GLD, XOM, etc.)

Data refreshes via cron (`scripts/data_refresh.py`), pair scans run weekly (`scripts/find_pairs.py`).

## Roadmap

- [x] Multi-source data pipeline
- [x] DuckDB storage
- [x] Cointegration-based pair discovery
- [x] Gymnasium trading environment
- [x] Feature engineering + TFT dataset builder
- [x] TFT spread predictor with quantile outputs
- [x] PPO agent training
- [x] Optuna hyperparameter tuning
- [x] Walk-forward backtesting with realistic costs
- [x] Streamlit dashboard
- [x] GitHub Actions CI + Docker deployment
- [ ] Live paper trading + signal alerts
- [ ] Modeling improvements to earn an edge over SPY (per-fold TFT, reward/feature work)

## License

[MIT](LICENSE) — see [Anjana Bandara](https://anjanamb.github.io/) for more projects.
