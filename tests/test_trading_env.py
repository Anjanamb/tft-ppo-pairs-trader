"""Tests for the pairs trading environment.

Covers Gymnasium API compliance and the exact mark-to-market reward arithmetic,
which is the part most likely to silently corrupt training if wrong.
"""

import numpy as np
import pytest

from src.agents.trading_env import PairsTradingEnv


def _cfg(cost=0.0, slippage=0.0, scaling=1.0, drawdown=0.0) -> dict:
    return {
        "ppo": {
            "transaction_cost_pct": cost,
            "slippage_pct": slippage,
            "reward_scaling": scaling,
            "drawdown_penalty": drawdown,
        }
    }


def test_gymnasium_api_compliance():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env

    rng = np.random.default_rng(0)
    spread = np.cumsum(rng.normal(0, 0.1, 200))
    env = PairsTradingEnv(spread, config=_cfg(), warmup=20)
    check_env(env)  # raises if the env violates the gym contract


def test_reward_is_mark_to_market():
    # spread chosen so steps are easy to verify by hand
    spread = np.array([0.0, 0.0, 1.0, 2.0, 1.5], dtype=float)
    env = PairsTradingEnv(spread, config=_cfg(scaling=1.0), warmup=2)
    env.reset()

    # go long at t=2: pnl = +1 * (spread[3]-spread[2]) = +1*(2-1) = 1
    _, r1, _, _, info1 = env.step(1)
    assert r1 == pytest.approx(1.0)
    assert info1["position"] == 1

    # go flat at t=3: position 0 over [3,4] earns nothing
    _, r2, term, _, info2 = env.step(0)
    assert r2 == pytest.approx(0.0)
    assert info2["position"] == 0
    assert term  # reached the penultimate index


def test_turnover_cost_is_charged():
    spread = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    env = PairsTradingEnv(spread, config=_cfg(cost=0.01, scaling=1.0), warmup=2)
    env.reset()
    # long: dspread = 0, but a position change of 1 costs 0.01
    _, r, _, _, info = env.step(1)
    assert r == pytest.approx(-0.01)
    assert info["n_trades"] == 1


def test_flat_policy_yields_zero_pnl():
    rng = np.random.default_rng(1)
    spread = np.cumsum(rng.normal(0, 0.1, 100))
    env = PairsTradingEnv(spread, config=_cfg(cost=0.01, scaling=1.0), warmup=20)
    env.reset()
    total = 0.0
    done = False
    while not done:
        _, r, term, trunc, info = env.step(0)  # always flat
        total += r
        done = term or trunc
    assert total == pytest.approx(0.0)
    assert info["n_trades"] == 0


def test_observation_shape_and_finiteness():
    spread = np.cumsum(np.random.default_rng(2).normal(0, 0.1, 80))
    forecast = spread + 0.05
    uncertainty = np.full_like(spread, 0.2)
    env = PairsTradingEnv(spread, forecast, uncertainty, config=_cfg(), warmup=20)
    obs, _ = env.reset()
    assert obs.shape == (9,)
    assert np.isfinite(obs).all()
    # forecast edge = forecast - spread = 0.05
    assert obs[4] == pytest.approx(0.05, abs=1e-5)
