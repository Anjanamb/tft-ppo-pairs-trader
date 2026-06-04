"""
Pairs Trading Gymnasium Environment.

The agent picks a *target position* on the spread each step and the environment
marks it to market:

    action 0 -> flat (0), 1 -> long spread (+1), 2 -> short spread (-1)

    reward_t = position_t * (spread_{t+1} - spread_t)            # mark-to-market
               - |position_t - position_{t-1}| * cost            # turnover cost
               - drawdown_penalty * new_drawdown

Everything is expressed in spread-return units, so costs and PnL are
commensurate and the undiscounted episode return equals realized PnL. Letting
the agent choose *flat* is deliberate: a mean-reversion strategy must be able to
exit to cash once the spread reverts, not only flip long/short.

The observation hands the agent the TFT's decision signals — the forecast edge
(median forecast minus current spread) and the forecast uncertainty
(q_90 - q_10) — alongside classic spread statistics.

References:
    - Han et al. (2023) "Select and Trade" — reward design
    - Stable-Baselines3 custom env pattern
"""

import logging

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.utils.config import load_config

logger = logging.getLogger(__name__)

# action index -> target position on the spread
_ACTION_TO_POSITION = {0: 0, 1: 1, 2: -1}


class PairsTradingEnv(gym.Env):
    """Single-pair spread trading environment with mark-to-market rewards."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        spread: np.ndarray,
        forecast: np.ndarray | None = None,
        uncertainty: np.ndarray | None = None,
        config: dict | None = None,
        warmup: int = 20,
    ):
        super().__init__()
        cfg = (config or load_config())["ppo"]

        self.spread = np.asarray(spread, dtype=np.float64)
        n = len(self.spread)
        self.forecast = (
            np.asarray(forecast, dtype=np.float64)
            if forecast is not None
            else self.spread.copy()  # naive: forecast == current (zero edge)
        )
        self.uncertainty = (
            np.asarray(uncertainty, dtype=np.float64)
            if uncertainty is not None
            else np.zeros(n)
        )

        self.cost = cfg["transaction_cost_pct"] + cfg["slippage_pct"]
        self.reward_scaling = cfg["reward_scaling"]
        self.drawdown_penalty = cfg["drawdown_penalty"]
        self.warmup = warmup

        # Precompute spread statistics once.
        self._zscore = self._rolling_zscore(self.spread, 20)
        self._ma_ratio = self._ma_ratio_arr(self.spread, 5, 20)
        self._vol = self._rolling_vol(self.spread, 20)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._reset_state()

    # ------------------------------------------------------------------
    def _reset_state(self):
        self.t = self.warmup
        self.position = 0
        self.cum_pnl = 0.0
        self.peak_pnl = 0.0
        self.n_trades = 0

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        prev_position = self.position
        target = _ACTION_TO_POSITION[int(action)]

        d_spread = self.spread[self.t + 1] - self.spread[self.t]
        turnover = abs(target - prev_position)
        cost = turnover * self.cost
        pnl = target * d_spread - cost

        self.position = target
        self.cum_pnl += pnl
        if turnover > 0:
            self.n_trades += 1

        # Drawdown bookkeeping (and optional penalty).
        self.peak_pnl = max(self.peak_pnl, self.cum_pnl)
        drawdown = self.peak_pnl - self.cum_pnl
        reward = (pnl - self.drawdown_penalty * drawdown) * self.reward_scaling

        self.t += 1
        terminated = self.t >= len(self.spread) - 1
        truncated = False

        info = {
            "position": self.position,
            "cum_pnl": self.cum_pnl,
            "drawdown": drawdown,
            "n_trades": self.n_trades,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        i = self.t
        horizon = len(self.spread) - 1 - self.warmup
        obs = np.array(
            [
                self.spread[i],
                self._zscore[i],
                self._ma_ratio[i],
                self._vol[i],
                self.forecast[i] - self.spread[i],   # forecast edge
                self.uncertainty[i],
                float(self.position),
                self.cum_pnl,
                (i - self.warmup) / max(horizon, 1),  # fraction of episode elapsed
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    @staticmethod
    def _rolling_zscore(data: np.ndarray, window: int) -> np.ndarray:
        out = np.zeros_like(data)
        for i in range(window, len(data)):
            w = data[i - window : i]
            out[i] = (data[i] - w.mean()) / (w.std() + 1e-8)
        return out

    @staticmethod
    def _ma_ratio_arr(data: np.ndarray, fast: int, slow: int) -> np.ndarray:
        out = np.zeros_like(data)
        for i in range(slow, len(data)):
            ma_fast = data[i - fast : i].mean()
            ma_slow = data[i - slow : i].mean()
            out[i] = ma_fast / (abs(ma_slow) + 1e-8)
        return out

    @staticmethod
    def _rolling_vol(data: np.ndarray, window: int) -> np.ndarray:
        out = np.zeros_like(data)
        for i in range(window, len(data)):
            out[i] = data[i - window : i].std()
        return out
