"""
Policy evaluation for the pairs trading environment.

Runs a policy through one episode and reports scale-invariant metrics, plus
baselines (flat, random, and a z-score mean-reversion rule). The z-score rule is
the one that matters: a learned agent that cannot beat a two-threshold rule has
not earned its complexity.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


def run_episode(env, policy_fn) -> dict:
    """
    Roll ``policy_fn(obs) -> action`` through one full episode.

    Returns per-step rewards/positions and the terminal info dict.
    """
    obs, _ = env.reset()
    rewards, positions = [], []
    info = {}
    done = False
    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        positions.append(info["position"])
        done = terminated or truncated
    return {"rewards": np.asarray(rewards), "positions": np.asarray(positions), "info": info}


def compute_metrics(rewards: np.ndarray, reward_scaling: float, info: dict) -> dict:
    """Sharpe / PnL / drawdown from per-step (scaled) rewards."""
    pnl = np.asarray(rewards, dtype=float) / reward_scaling  # back to return units
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity) if len(equity) else np.array([0.0])
    max_dd = float((peak - equity).max()) if len(equity) else 0.0
    sharpe = float(pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(_TRADING_DAYS))
    return {
        "total_pnl": float(pnl.sum()),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": int(info.get("n_trades", 0)),
        "n_steps": len(pnl),
    }


def evaluate(env, policy_fn, reward_scaling: float) -> dict:
    """Convenience: run one episode and score it."""
    ep = run_episode(env, policy_fn)
    return compute_metrics(ep["rewards"], reward_scaling, ep["info"])


# ----------------------------------------------------------------------
# Baseline policies (obs layout defined in trading_env._get_obs)
# ----------------------------------------------------------------------
def flat_policy(_obs) -> int:
    return 0


def random_policy(env):
    return lambda _obs: int(env.action_space.sample())


def zscore_policy(entry: float = 1.0, exit_band: float = 0.5):
    """Classic mean reversion: fade large z-scores, flatten near the mean."""

    def policy(obs) -> int:
        z = float(obs[1])  # spread z-score
        if z > entry:
            return 2       # spread rich -> short
        if z < -entry:
            return 1       # spread cheap -> long
        if abs(z) < exit_band:
            return 0       # reverted -> flat
        return 0

    return policy


def regime_zscore_policy(
    entry: float = 1.0,
    exit_band: float = 0.5,
    window: int = 40,
    max_half_life: float = 60.0,
):
    """
    Z-score rule with a regime gate: only take a position while the spread is
    *currently* mean-reverting.

    A rolling OU half-life is estimated on the last ``window`` spreads the policy
    has seen; if the spread is not mean-reverting (no finite half-life) or reverts
    too slowly (> ``max_half_life`` days), the relationship has likely broken down
    and the policy stays flat instead of fading a runaway divergence.

    Stateful: it buffers the spreads streamed through ``obs[0]``, so a fresh
    policy must be created per backtest fold (the strategy factory does this).
    """
    from src.pairs.selector import PairSelector

    buffer: list[float] = []

    def policy(obs) -> int:
        buffer.append(float(obs[0]))     # spread level
        if len(buffer) > window:
            buffer.pop(0)
        z = float(obs[1])

        base = 2 if z > entry else 1 if z < -entry else 0
        if base == 0:
            return 0

        # Regime gate — needs enough history to estimate a half-life.
        if len(buffer) >= 20:
            hl = PairSelector._compute_half_life(pd.Series(buffer))
            if hl is None or hl > max_half_life:
                return 0  # not mean-reverting now -> stand aside
        return base

    return policy


def build_env_inputs(
    pair_panel: pd.DataFrame, forecasts: pd.DataFrame | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align a single pair's spread with per-step TFT forecasts for the env.

    Args:
        pair_panel: rows for one pair with ``time_idx`` and ``spread``.
        forecasts: optional frame with ``time_idx``, ``prediction``,
            ``uncertainty`` (1-step-ahead). Missing rows fall back to a zero-edge
            naive forecast.

    Returns:
        ``(spread, forecast, uncertainty)`` arrays ordered by time_idx.
    """
    panel = pair_panel.sort_values("time_idx")[["time_idx", "spread"]]
    if forecasts is not None and not forecasts.empty:
        panel = panel.merge(
            forecasts[["time_idx", "prediction", "uncertainty"]],
            on="time_idx",
            how="left",
        )
        forecast = panel["prediction"].fillna(panel["spread"]).to_numpy()
        med_unc = panel["uncertainty"].median()
        uncertainty = panel["uncertainty"].fillna(med_unc).to_numpy()
    else:
        forecast = panel["spread"].to_numpy()
        uncertainty = np.zeros(len(panel))
    return panel["spread"].to_numpy(), forecast, uncertainty
