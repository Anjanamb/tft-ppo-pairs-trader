"""
Optuna hyperparameter optimization for the TFT + PPO pipeline.

Two targets:

* ``ppo`` (default) — fix the TFT forecasts (loaded once) and tune the agent
  against **out-of-sample** Sharpe on a held-out window. Optimizing OOS, not
  in-sample, is the whole point: it stops the search from rewarding overfit
  configs. Intermediate OOS Sharpe is reported during training so the
  MedianPruner can kill weak trials early.
* ``tft`` — tune the forecaster against validation quantile loss.

Runs are persisted to a SQLite study (resumable) and logged to MLflow. There is
no Claude Code routine — this is a plain CLI:

    python -m src.tuning.optimizer --target ppo --tft models/tft_20260604.ckpt \
        --n-trials 50 --timesteps 150000
"""

import argparse
import copy
import logging
import sys
from datetime import datetime
from pathlib import Path

import optuna
import pandas as pd
import yaml
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.evaluation import build_env_inputs, evaluate, run_episode, compute_metrics
from src.agents.ppo_agent import TradingAgent
from src.agents.trading_env import PairsTradingEnv
from src.data.manager import DataManager
from src.models.dataset import TFTDatasetBuilder
from src.tuning.search_space import ppo_search_space, tft_search_space
from src.utils.config import load_config

logger = logging.getLogger("optimizer")


class SharpePruneCallback(BaseCallback):
    """Periodically score OOS Sharpe and let Optuna prune weak trials."""

    def __init__(self, test_env, scaling: float, trial, eval_freq: int):
        super().__init__()
        self.test_env = test_env
        self.scaling = scaling
        self.trial = trial
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        def policy(obs):
            return int(self.model.predict(obs, deterministic=True)[0])

        ep = run_episode(self.test_env, policy)
        sharpe = compute_metrics(ep["rewards"], self.scaling, ep["info"])["sharpe"]
        self.trial.report(sharpe, self.num_timesteps)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return True


# ----------------------------------------------------------------------
def _prepare_arrays(cfg: dict, row: pd.DataFrame, tft_path: Path | None):
    """Spread + forecast + uncertainty arrays for one pair (forecasts fixed)."""
    panel = TFTDatasetBuilder(cfg).build_panel(row, DataManager(cfg))
    if panel.empty:
        raise RuntimeError("Empty feature panel for the requested pair.")
    forecasts = None
    if tft_path is not None:
        from src.models.tft_predictor import TFTPredictor

        predictor = TFTPredictor(cfg)
        predictor.load(tft_path)
        forecasts = predictor.predict_per_step(panel)
    return build_env_inputs(panel, forecasts)


def make_ppo_objective(cfg, arrays, test_fraction, timesteps, eval_freq):
    spread, forecast, uncertainty = arrays
    split = int(len(spread) * (1 - test_fraction))
    seed = cfg["tft"]["seed"]
    scaling = cfg["ppo"]["reward_scaling"]

    def objective(trial) -> float:
        tcfg = copy.deepcopy(cfg)
        tcfg["ppo"].update(ppo_search_space(trial))
        train_env = PairsTradingEnv(
            spread[:split], forecast[:split], uncertainty[:split], config=tcfg
        )
        test_env = PairsTradingEnv(
            spread[split:], forecast[split:], uncertainty[split:], config=tcfg
        )
        agent = TradingAgent(tcfg)
        cb = SharpePruneCallback(test_env, scaling, trial, eval_freq)
        agent.train(train_env, total_timesteps=timesteps, seed=seed, callback=cb)
        return evaluate(test_env, lambda o: agent.predict(o), scaling)["sharpe"]

    return objective


def make_tft_objective(cfg, pairs_df, top, epochs):
    def objective(trial) -> float:
        tcfg = copy.deepcopy(cfg)
        tcfg["tft"].update(tft_search_space(trial))
        tcfg["tft"]["max_epochs"] = epochs
        from src.models.tft_predictor import TFTPredictor

        builder = TFTDatasetBuilder(tcfg)
        panel = builder.build_panel(pairs_df.head(top), DataManager(tcfg))
        training, validation = builder.make_datasets(panel)
        metrics = TFTPredictor(tcfg).train(training, validation)
        val = metrics["best_val_loss"]
        return float(val) if val is not None else float("inf")

    return objective


def _build_study(cfg: dict, target: str):
    tcfg = cfg["tuning"]
    direction = "maximize" if target == "ppo" else "minimize"
    sampler = optuna.samplers.TPESampler(seed=cfg["tft"]["seed"])
    pruner = (
        optuna.pruners.MedianPruner()
        if tcfg.get("pruner") == "median"
        else optuna.pruners.NopPruner()
    )
    return optuna.create_study(
        study_name=f"{tcfg['study_name']}_{target}",
        storage=tcfg["storage"],
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def _log_trials_to_mlflow(study, cfg, target):
    import mlflow

    mlflow.set_tracking_uri(cfg["logging"]["mlflow"]["tracking_uri"])
    mlflow.set_experiment(f"{cfg['logging']['mlflow']['experiment_name']}_{target}")
    for t in study.trials:
        if t.value is None:
            continue
        with mlflow.start_run(run_name=f"trial_{t.number}"):
            mlflow.log_params(t.params)
            mlflow.log_metric("objective", t.value)


def run_study(target="ppo", pair=None, tft_path=None, n_trials=None,
              timesteps=30000, timeout=None, top=3, tft_epochs=15,
              test_fraction=0.2, log_mlflow=True, config=None) -> optuna.Study:
    cfg = config or load_config()
    n_trials = n_trials or cfg["tuning"]["n_trials"]

    pairs_df = pd.read_csv(sorted(Path("data/pairs").glob("pairs_*.csv"))[-1])

    if target == "ppo":
        row = pairs_df[pairs_df["pair_id"] == pair] if pair else pairs_df.head(1)
        arrays = _prepare_arrays(cfg, row, tft_path)
        eval_freq = max(timesteps // 5, 1)
        objective = make_ppo_objective(
            cfg, arrays, test_fraction, timesteps, eval_freq
        )
    elif target == "tft":
        objective = make_tft_objective(cfg, pairs_df, top, tft_epochs)
    else:
        raise ValueError(f"Unknown target '{target}' (use 'ppo' or 'tft')")

    study = _build_study(cfg, target)
    logger.info("Optimizing %s - %d trials (timeout=%s)", target, n_trials, timeout)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info("Best %s value: %.4f", target, study.best_value)
    logger.info("Best params: %s", study.best_params)
    _save_best_params(study, target)
    if log_mlflow:
        # Logging is a nice-to-have — never let it discard a finished study.
        try:
            _log_trials_to_mlflow(study, cfg, target)
        except Exception as exc:
            logger.warning("MLflow logging skipped (%s)", exc)
    return study


def _save_best_params(study, target: str):
    out = Path("configs/best_params.yaml")
    payload = {}
    if out.exists():
        payload = yaml.safe_load(out.read_text()) or {}
    payload[target] = {
        "best_value": float(study.best_value),
        "params": study.best_params,
        "n_trials": len(study.trials),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    out.write_text(yaml.safe_dump(payload, sort_keys=False))
    logger.info("Saved best %s params to %s", target, out)


def main():
    parser = argparse.ArgumentParser(description="Tune the TFT+PPO pipeline.")
    parser.add_argument("--target", choices=["ppo", "tft"], default="ppo")
    parser.add_argument("--pair", type=str, default=None)
    parser.add_argument("--tft", type=Path, default=None,
                        help="TFT checkpoint (default: latest models/tft_*.ckpt)")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=30000,
                        help="PPO timesteps per trial")
    parser.add_argument("--timeout", type=int, default=None,
                        help="overall study timeout in seconds")
    parser.add_argument("--top", type=int, default=3,
                        help="pairs to train on for --target tft")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/optimizer.log", mode="a", encoding="utf-8"),
        ],
    )
    from src.utils.runtime import configure_quiet_runtime
    configure_quiet_runtime()

    tft_path = args.tft
    if tft_path is None and args.target == "ppo":
        from src.models.tft_predictor import find_latest_checkpoint

        tft_path = find_latest_checkpoint()
        if tft_path:
            logger.info("Auto-selected latest TFT checkpoint: %s", tft_path)
    elif tft_path is not None and not tft_path.exists():
        logger.error("TFT checkpoint not found: %s", tft_path)
        sys.exit(1)

    run_study(
        target=args.target, pair=args.pair, tft_path=tft_path,
        n_trials=args.n_trials, timesteps=args.timesteps, timeout=args.timeout,
        top=args.top, log_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
