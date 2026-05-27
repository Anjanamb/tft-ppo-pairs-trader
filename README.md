# tft-ppo-pairs-trader

Multi-asset pairs trading with TFT spread prediction and PPO execution.

[![Status](https://img.shields.io/badge/status-active%20development-yellow?style=flat-square)](#roadmap)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=flat-square&logo=duckdb&logoColor=black)](https://duckdb.org/)

The idea: use a [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363) to forecast the spread between cointegrated asset pairs, then let a PPO agent decide when to enter/exit trades based on those forecasts + uncertainty estimates. Runs across crypto (Binance), US equities, ETFs, and commodities.

> **Status:** Core infra (data pipeline, pair discovery, trading env) is done. Active work: TFT predictor, PPO training loop, walk-forward backtester. See [Roadmap](#roadmap) for the full list.

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

# PyTorch first — pick your hardware
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # NVIDIA
# or: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install pytorch-forecasting pytorch-lightning
pip install stable-baselines3 gymnasium
pip install -r requirements.txt
```

## Usage

```bash
# Pull 5 years of data for all configured tickers
python scripts/data_refresh.py

# Scan for cointegrated pairs
python scripts/find_pairs.py
```

## What's in the box

```
configs/config.yaml        <- all tickers, model params, schedules
src/data/                  <- data sources (yfinance, ccxt) + DuckDB manager
src/pairs/selector.py      <- cointegration tests, half-life, pair ranking
src/models/                <- TFT predictor (base interface + implementation)
src/agents/trading_env.py  <- Gymnasium env for the PPO agent
src/backtest/              <- walk-forward backtesting
src/dashboard/             <- Streamlit app
scripts/                   <- cron jobs (data refresh, pair scan, backtest)
```

## Initial results

First pair scan across 26 assets (325 combinations):

```
BNB/USDT ↔ XLF     corr=0.888  coint_p=0.010  half_life=21.0d  score=0.892
GC=F     ↔ GS      corr=0.967  coint_p=0.006  half_life=33.4d  score=0.839
GLD      ↔ GS      corr=0.966  coint_p=0.006  half_life=34.5d  score=0.831
JPM      ↔ SPY     corr=0.984  coint_p=0.021  half_life=28.5d  score=0.784
AAPL     ↔ NVDA    corr=0.933  coint_p=0.008  half_life=39.8d  score=0.756
```

12 valid pairs total, including cross-asset (crypto ↔ equity) and negatively correlated pairs.

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
- [ ] Feature engineering + TFT dataset builder
- [ ] TFT spread predictor with quantile outputs
- [ ] PPO agent training
- [ ] Optuna hyperparameter tuning
- [ ] Walk-forward backtesting with realistic costs
- [ ] Streamlit dashboard
- [ ] Live paper trading + signal alerts
- [ ] Docker deployment

## License

[MIT](LICENSE) — see [Anjana Bandara](https://anjanamb.github.io/) for more projects.
