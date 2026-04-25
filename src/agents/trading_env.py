"""
Pairs Trading Gymnasium Environment.

The agent observes spread dynamics + TFT predictions and decides:
  - Action space: Discrete(5) or Box (continuous)
    0: Hold, 1: Long spread, 2: Short spread, 3: Close long, 4: Close short

Reward: risk-adjusted PnL with transaction cost penalties.

References:
    - Han et al. (2023) "Select and Trade" — reward design
    - Stable-Baselines3 custom env pattern
"""

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.utils.config import load_config

logger = logging.getLogger(__name__)


class PairsTradingEnv(gym.Env):
    """
    Custom Gymnasium environment for pairs spread trading.

    Observation space:
        - spread (current)
        - spread z-score
        - spread moving average ratio
        - volatility ratio
        - TFT prediction (median)
        - TFT prediction uncertainty (q_high - q_low)
        - current position (-1, 0, 1)
        - unrealized PnL
        - time features (day_of_week, hour, etc.)

    Action space:
        Discrete(3): 0=hold, 1=long_spread, 2=short_spread
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        spread_data: np.ndarray,
        predictions: np.ndarray | None = None,
        pred_uncertainty: np.ndarray | None = None,
        config: dict | None = None,
    ):
        super().__init__()

        cfg = (config or load_config())["ppo"]

        self.spread = spread_data
        self.predictions = predictions if predictions is not None else np.zeros_like(spread_data)
        self.pred_uncertainty = pred_uncertainty if pred_uncertainty is not None else np.ones_like(spread_data)

        self.initial_balance = cfg["initial_balance"]
        self.transaction_cost = cfg["transaction_cost_pct"]
        self.slippage = cfg["slippage_pct"]
        self.max_position = cfg["max_position_size"]

        # Spaces
        n_features = 9  # spread, zscore, ma_ratio, vol, pred, uncert, pos, pnl, time
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # hold, long, short

        # State
        self.current_step = 0
        self.position = 0  # -1, 0, 1
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.total_pnl = 0.0
        self.trades = []

        # Precompute features
        self._zscore = self._compute_zscore(self.spread, window=20)
        self._ma_ratio = self._compute_ma_ratio(self.spread, fast=5, slow=20)
        self._volatility = self._compute_rolling_vol(self.spread, window=20)

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 20  # skip warmup period
        self.position = 0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.total_pnl = 0.0
        self.trades = []
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        prev_balance = self.balance
        current_spread = self.spread[self.current_step]

        # Execute action
        reward = 0.0
        cost = 0.0

        if action == 1 and self.position <= 0:  # Go long spread
            if self.position == -1:  # Close short first
                pnl = (self.entry_price - current_spread)
                cost = abs(current_spread) * (self.transaction_cost + self.slippage)
                self.balance += pnl - cost
            # Open long
            self.position = 1
            self.entry_price = current_spread
            cost += abs(current_spread) * (self.transaction_cost + self.slippage)
            self.balance -= cost

        elif action == 2 and self.position >= 0:  # Go short spread
            if self.position == 1:  # Close long first
                pnl = (current_spread - self.entry_price)
                cost = abs(current_spread) * (self.transaction_cost + self.slippage)
                self.balance += pnl - cost
            # Open short
            self.position = -1
            self.entry_price = current_spread
            cost += abs(current_spread) * (self.transaction_cost + self.slippage)
            self.balance -= cost

        elif action == 0 and self.position != 0:  # Hold/close
            pass  # just hold existing position

        # Compute unrealized PnL
        if self.position == 1:
            unrealized = current_spread - self.entry_price
        elif self.position == -1:
            unrealized = self.entry_price - current_spread
        else:
            unrealized = 0.0

        # Reward: change in portfolio value, penalize transaction costs
        portfolio_value = self.balance + unrealized
        reward = (portfolio_value - prev_balance) / self.initial_balance

        # Advance
        self.current_step += 1
        terminated = self.current_step >= len(self.spread) - 1
        truncated = False

        # Track trade
        if cost > 0:
            self.trades.append({
                "step": self.current_step,
                "action": action,
                "spread": current_spread,
                "position": self.position,
                "cost": cost,
                "balance": self.balance,
            })

        info = {
            "balance": self.balance,
            "position": self.position,
            "unrealized_pnl": unrealized,
            "n_trades": len(self.trades),
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        i = self.current_step
        obs = np.array([
            self.spread[i],
            self._zscore[i],
            self._ma_ratio[i],
            self._volatility[i],
            self.predictions[i],
            self.pred_uncertainty[i],
            float(self.position),
            (self.balance - self.initial_balance) / self.initial_balance,
            float(i % 5) / 5.0,  # day_of_week normalized
        ], dtype=np.float32)

        # Replace NaN/inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    @staticmethod
    def _compute_zscore(data: np.ndarray, window: int = 20) -> np.ndarray:
        result = np.zeros_like(data)
        for i in range(window, len(data)):
            w = data[i - window : i]
            mean, std = w.mean(), w.std()
            result[i] = (data[i] - mean) / (std + 1e-8)
        return result

    @staticmethod
    def _compute_ma_ratio(data: np.ndarray, fast: int = 5, slow: int = 20) -> np.ndarray:
        result = np.zeros_like(data)
        for i in range(slow, len(data)):
            ma_fast = data[i - fast : i].mean()
            ma_slow = data[i - slow : i].mean()
            result[i] = ma_fast / (ma_slow + 1e-8)
        return result

    @staticmethod
    def _compute_rolling_vol(data: np.ndarray, window: int = 20) -> np.ndarray:
        result = np.zeros_like(data)
        for i in range(window, len(data)):
            result[i] = data[i - window : i].std()
        return result
