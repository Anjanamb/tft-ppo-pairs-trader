"""
Optuna search spaces for the TFT and PPO components.

Each function takes an Optuna ``Trial`` and returns a dict of hyperparameters
that overrides the matching block in ``config.yaml``. The two are tuned against
different objectives — TFT against validation quantile loss, PPO against
out-of-sample Sharpe — so they live in separate spaces.
"""

from __future__ import annotations


def ppo_search_space(trial) -> dict:
    """PPO hyperparameters. batch_size always divides n_steps (n_envs=1)."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "ent_coef": trial.suggest_float("ent_coef", 1e-3, 1e-1, log=True),
    }


def tft_search_space(trial) -> dict:
    """
    TFT hyperparameters. hidden_continuous_size is capped at hidden_size, which
    pytorch-forecasting requires (continuous embeddings feed the main width).
    """
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    return {
        "hidden_size": hidden_size,
        "attention_head_size": trial.suggest_categorical(
            "attention_head_size", [1, 2, 4]
        ),
        "dropout": trial.suggest_float("dropout", 0.05, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "hidden_continuous_size": min(
            trial.suggest_int("hidden_continuous_size", 8, 64), hidden_size
        ),
    }
