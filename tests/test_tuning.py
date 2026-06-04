"""Tests for the Optuna tuning pipeline."""

import copy

import numpy as np
import pytest

from src.tuning.search_space import ppo_search_space, tft_search_space
from src.utils.config import load_config


def test_ppo_search_space_keys_and_ranges():
    optuna = pytest.importorskip("optuna")
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

    seen = []

    def objective(trial):
        p = ppo_search_space(trial)
        seen.append(p)
        assert set(p) == {
            "learning_rate", "n_steps", "batch_size", "gamma",
            "gae_lambda", "clip_range", "ent_coef",
        }
        assert p["n_steps"] in (1024, 2048, 4096)
        assert p["batch_size"] in (32, 64, 128)
        assert 0.95 <= p["gamma"] <= 0.999
        return 0.0

    study.optimize(objective, n_trials=5)
    assert len(seen) == 5


def test_tft_search_space_caps_continuous_size():
    optuna = pytest.importorskip("optuna")
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))

    def objective(trial):
        p = tft_search_space(trial)
        # continuous size must never exceed hidden_size
        assert p["hidden_continuous_size"] <= p["hidden_size"]
        assert p["attention_head_size"] in (1, 2, 4)
        return 0.0

    study.optimize(objective, n_trials=8)


def _mean_reverting(n=160, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 0.1)
    return x


def test_ppo_study_runs_end_to_end():
    pytest.importorskip("optuna")
    pytest.importorskip("stable_baselines3")
    from src.tuning.optimizer import make_ppo_objective, _build_study

    cfg = copy.deepcopy(load_config())
    cfg["ppo"].update(reward_scaling=100.0)
    spread = _mean_reverting()
    arrays = (spread, spread.copy(), np.zeros_like(spread))

    objective = make_ppo_objective(
        cfg, arrays, test_fraction=0.25, timesteps=256, eval_freq=128
    )
    import optuna

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)

    assert len(study.trials) == 2
    assert study.best_value is not None
