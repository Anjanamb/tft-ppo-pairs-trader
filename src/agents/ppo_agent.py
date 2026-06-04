"""
RL trading agent wrapper around Stable-Baselines3.

The algorithm is chosen by ``ppo.algorithm`` in config (PPO, A2C, SAC, ...).
Hyperparameters are filtered against the chosen algorithm's constructor, so
swapping PPO -> SAC does not blow up on PPO-only kwargs like ``clip_range`` —
this is what makes the "change one config line" design decision actually hold.
"""

import inspect
import logging
from pathlib import Path

import numpy as np

from src.utils.config import load_config

logger = logging.getLogger(__name__)

# PPO/A2C-style kwargs we try to pass; each is kept only if the algorithm
# actually accepts it.
_CANDIDATE_KWARGS = [
    "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
    "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
]


class TradingAgent:
    """SB3 policy wrapper: train / predict / persist."""

    def __init__(self, config: dict | None = None):
        self.cfg = (config or load_config())["ppo"]
        self.algo_name = self.cfg["algorithm"]
        self.model = None

    def _algo_cls(self):
        import stable_baselines3 as sb3

        try:
            return getattr(sb3, self.algo_name)
        except AttributeError as exc:
            raise ValueError(
                f"Unknown SB3 algorithm '{self.algo_name}'. "
                f"Available: PPO, A2C, DDPG, SAC, TD3, DQN."
            ) from exc

    def _supported_kwargs(self, algo_cls) -> dict:
        params = inspect.signature(algo_cls.__init__).parameters
        kwargs = {k: self.cfg[k] for k in _CANDIDATE_KWARGS if k in self.cfg and k in params}
        dropped = [k for k in _CANDIDATE_KWARGS if k in self.cfg and k not in params]
        if dropped:
            logger.info("%s ignores kwargs: %s", self.algo_name, dropped)
        return kwargs

    def train(self, env, total_timesteps: int | None = None,
              seed: int | None = None, callback=None):
        """Fit the policy on ``env``. Returns self for chaining."""
        algo_cls = self._algo_cls()
        kwargs = self._supported_kwargs(algo_cls)
        steps = total_timesteps or self.cfg["total_timesteps"]

        self.model = algo_cls(
            self.cfg["policy"],
            env,
            seed=seed,
            device=self.cfg.get("device", "cpu"),
            verbose=0,
            **kwargs,
        )
        logger.info(
            "Training %s for %d timesteps (%s)",
            self.algo_name, steps, self.cfg["policy"],
        )
        self.model.learn(total_timesteps=steps, callback=callback, progress_bar=False)
        return self

    def predict(self, obs, deterministic: bool = True):
        if self.model is None:
            raise RuntimeError("Agent is not trained or loaded.")
        action, _ = self.model.predict(np.asarray(obs), deterministic=deterministic)
        return int(action)

    def save(self, path: Path):
        if self.model is None:
            raise RuntimeError("Nothing to save — agent is not trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info("Saved %s agent to %s", self.algo_name, path)

    def load(self, path: Path, env=None):
        algo_cls = self._algo_cls()
        self.model = algo_cls.load(str(Path(path)), env=env)
        logger.info("Loaded %s agent from %s", self.algo_name, path)
        return self
