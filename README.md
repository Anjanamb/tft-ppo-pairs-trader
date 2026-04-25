# TFT + PPO Multi-Asset Pairs Trading System

A production-grade pairs trading system that combines **Temporal Fusion Transformer (TFT)** for spread forecasting with **Proximal Policy Optimization (PPO)** for trade execution — across crypto, equities, ETFs, and commodities.

> **Why this project?** Academic research has validated TFT for financial forecasting and RL for trade execution independently. Multiple papers explicitly call out their integration as future work. This project builds the end-to-end system those papers envision.

## Academic Foundation

| Paper | Contribution | Code? |
|-------|-------------|-------|
| Lim et al. (2021) "[Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363)" | TFT architecture for multi-horizon forecasting | ✅ (Google) |
| Han et al. (2023) "[Select and Trade](https://arxiv.org/abs/2301.10724)" | Hierarchical RL for unified pair selection + trading | ✅ (limited) |
| Peik et al. (2025) "[Adaptive TFT for Crypto](https://arxiv.org/abs/2509.10542)" | Adaptive TFT with dynamic segmentation; flags RL as future work | ❌ |
| DCN (2025) "[Hierarchical DL for Pair Trading with GAT + A2C](https://www.sciencedirect.com/science/article/pii/S0957417425021153)" | Graph attention for pair selection + DRL for execution | ❌ |

**This project fills the gap:** a working implementation of TFT + PPO for multi-asset pairs trading with automated HP tuning, backtesting, and a live dashboard.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    config.yaml                          │
│         (tickers, params, schedules — one file)         │
└────────────────────────┬────────────────────────────────┘
                         │
    ┌──────────┬─────────┼──────────┬───────────┐
    ▼          ▼         ▼          ▼           ▼
 Crypto     ETFs     Stocks    Commodities   Macro
 (Binance)  (yf)     (yf)      (yf)         (FRED)
    └──────────┴─────────┼──────────┴───────────┘
                         ▼
              ┌─────────────────────┐
              │  Pair Selection     │
              │  (cointegration,    │
              │   correlation,      │
              │   half-life)        │
              └─────────┬───────────┘
                        ▼
         ┌──────────────┴──────────────┐
         ▼                             ▼
  ┌──────────────┐           ┌──────────────────┐
  │ TFT Predictor│──────────▶│   PPO Agent      │
  │ (spread +    │ forecast  │ (entry/exit/     │
  │  uncertainty)│           │  position sizing)│
  └──────────────┘           └────────┬─────────┘
                                      │
              ┌───────────┬───────────┼───────────┐
              ▼           ▼           ▼           ▼
         Backtester  Paper Trader  Signals   Dashboard
         (Sharpe,    (live sim)    (alerts)  (Streamlit)
          costs)
```

## Project Structure

```
tft-ppo-pairs-trader/
├── configs/
│   └── config.yaml          # Single source of truth
├── src/
│   ├── data/
│   │   ├── base.py          # BaseDataSource interface
│   │   ├── yfinance_source.py
│   │   ├── crypto_source.py
│   │   └── manager.py       # DataManager (DuckDB)
│   ├── pairs/
│   │   └── selector.py      # Cointegration-based pair discovery
│   ├── models/
│   │   └── base.py          # BasePredictor interface
│   ├── agents/
│   │   └── trading_env.py   # Gymnasium env for PPO
│   ├── backtest/            # Walk-forward backtesting
│   ├── dashboard/           # Streamlit app
│   └── utils/
│       └── config.py        # Config loader
├── scripts/
│   ├── data_refresh.py      # Cron: every 6 hours
│   └── find_pairs.py        # Cron: weekly
├── notebooks/               # Exploration & analysis
├── tests/
├── data/                    # DuckDB, CSVs, model artifacts
├── logs/
└── requirements.txt
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/Anjanamb/tft-ppo-pairs-trader.git
cd tft-ppo-pairs-trader
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Initialize database and fetch data
python scripts/data_refresh.py

# Scan for cointegrated pairs
python scripts/find_pairs.py
```

## Extensibility

Adding a new asset = one line in `config.yaml`:
```yaml
stocks:
  tickers:
    - "TSLA"  # just add this
```

Swapping the RL algorithm = one line:
```yaml
ppo:
  algorithm: "SAC"  # or A2C, TD3
```

Adding a new predictor = implement `BasePredictor`:
```python
class MyModel(BasePredictor):
    def train(self, train_data, val_data): ...
    def predict(self, data): ...
```

## Automation

| Task | Method | Schedule |
|------|--------|----------|
| Data refresh | Cron + Python | Every 6 hours |
| Pair scan | Cron + Python | Weekly (Sunday) |
| Backtest | Cron + Python | Weekly (Sunday) |
| HP tuning | Claude Code routine | On demand |
| Report gen | Claude Code routine | On demand |
| Code review | Claude Code routine | On demand |

## Roadmap

- [x] Project scaffold & config
- [x] Multi-source data pipeline (yfinance, ccxt)
- [x] DuckDB storage layer
- [x] Cointegration-based pair selector
- [x] Gymnasium trading environment
- [ ] TFT spread predictor (pytorch-forecasting)
- [ ] PPO agent training (stable-baselines3)
- [ ] Optuna HP tuning pipeline
- [ ] Walk-forward backtester
- [ ] Streamlit dashboard
- [ ] Live paper trading
- [ ] Signal alerts (email/Telegram)
- [ ] Docker deployment
- [ ] CI/CD with GitHub Actions

## License

MIT
