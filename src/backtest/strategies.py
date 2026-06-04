"""
Backtest strategies.

A strategy is a callable ``fit(train_env, cfg) -> policy_fn``. The walk-forward
engine calls it once per fold with an environment built from the *train* window
only, so any learning happens on past data alone. The returned ``policy_fn`` is
then rolled through the held-out test window.
"""

import logging

from src.agents.evaluation import zscore_policy
from src.agents.ppo_agent import TradingAgent

logger = logging.getLogger(__name__)


def zscore_strategy(entry: float = 1.0, exit_band: float = 0.5):
    """Stateless mean-reversion rule — nothing is fit, so it is pure OOS."""

    def fit(train_env, cfg):
        return zscore_policy(entry, exit_band)

    return fit


def ppo_strategy(timesteps: int = 30000, seed: int = 42):
    """Retrain a PPO agent on each fold's train window."""

    def fit(train_env, cfg):
        agent = TradingAgent(cfg).train(train_env, total_timesteps=timesteps, seed=seed)

        def policy(obs):
            return agent.predict(obs, deterministic=True)

        return policy

    return fit
