"""Tests for the SB3 trading agent wrapper and evaluation helpers."""

import copy

import numpy as np
import pytest

from src.agents.evaluation import (
    build_env_inputs,
    compute_metrics,
    evaluate,
    flat_policy,
    zscore_policy,
)
from src.agents.ppo_agent import TradingAgent
from src.agents.trading_env import PairsTradingEnv
from src.utils.config import load_config

import pandas as pd


def _agent_config(algorithm="PPO") -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["ppo"].update(
        algorithm=algorithm,
        n_steps=64,
        batch_size=32,
        total_timesteps=128,
        reward_scaling=100.0,
    )
    return cfg


def _spread(n=150, seed=0):
    rng = np.random.default_rng(seed)
    # mean-reverting (Ornstein-Uhlenbeck-ish) so there is signal to trade
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 0.1)
    return x


def test_a2c_swap_drops_ppo_only_kwargs():
    # PPO-only kwargs (clip_range, n_epochs, batch_size) must not be forwarded
    # to A2C, otherwise construction would raise.
    import stable_baselines3 as sb3

    agent = TradingAgent(_agent_config("A2C"))
    supported = agent._supported_kwargs(sb3.A2C)
    assert "clip_range" not in supported
    assert "n_epochs" not in supported
    assert "learning_rate" in supported


def test_train_predict_save_load(tmp_path):
    pytest.importorskip("stable_baselines3")

    cfg = _agent_config("PPO")
    env = PairsTradingEnv(_spread(), config=cfg, warmup=20)

    agent = TradingAgent(cfg).train(env, seed=0)
    obs, _ = env.reset()
    action = agent.predict(obs)
    assert action in (0, 1, 2)

    ckpt = tmp_path / "agent.zip"
    agent.save(ckpt)
    assert ckpt.exists()

    reloaded = TradingAgent(cfg).load(ckpt)
    assert reloaded.predict(obs) == action  # deterministic policy reproducible


def test_metrics_and_baselines_run():
    cfg = _agent_config()
    env = PairsTradingEnv(_spread(seed=3), config=cfg, warmup=20)

    flat = evaluate(env, flat_policy, cfg["ppo"]["reward_scaling"])
    assert flat["total_pnl"] == pytest.approx(0.0)
    assert flat["n_trades"] == 0

    rule = evaluate(env, zscore_policy(), cfg["ppo"]["reward_scaling"])
    assert set(rule.keys()) == {"total_pnl", "sharpe", "max_drawdown", "n_trades", "n_steps"}
    assert rule["n_steps"] > 0


def test_build_env_inputs_aligns_and_fills():
    panel = pd.DataFrame({"time_idx": [0, 1, 2, 3], "spread": [0.1, 0.2, 0.3, 0.4]})
    forecasts = pd.DataFrame(
        {"time_idx": [2, 3], "prediction": [0.35, 0.45], "uncertainty": [0.1, 0.2]}
    )
    spread, forecast, unc = build_env_inputs(panel, forecasts)

    assert np.allclose(spread, [0.1, 0.2, 0.3, 0.4])
    # missing early forecasts fall back to the spread itself (zero edge)
    assert np.allclose(forecast, [0.1, 0.2, 0.35, 0.45])
    # missing uncertainty filled with the median of observed (median of .1,.2)
    assert np.allclose(unc, [0.15, 0.15, 0.1, 0.2])


def test_compute_metrics_drawdown():
    # equity rises then falls: pnl scaled by reward_scaling=10
    rewards = np.array([10.0, 10.0, -20.0])  # /10 -> [1, 1, -2]; equity [1,2,0]
    m = compute_metrics(rewards, 10.0, {"n_trades": 2})
    assert m["total_pnl"] == pytest.approx(0.0)
    assert m["max_drawdown"] == pytest.approx(2.0)  # peak 2 -> 0
    assert m["n_trades"] == 2
