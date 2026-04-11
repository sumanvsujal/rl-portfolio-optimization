"""Gymnasium environment for long-only portfolio rebalancing."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from .reward import RewardConfig, cvar_adjusted_sharpe_reward


class PortfolioEnv(gym.Env[np.ndarray, np.ndarray]):
    """Portfolio rebalancing environment used by the RL agents.

    The environment is long-only and projects every action back onto the
    simplex of valid portfolio weights. This is deliberate: the research
    question is about autonomous portfolio allocation, not about leverage
    or short selling.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: pd.DataFrame,
        features: pd.DataFrame,
        lookback_window: int = 20,
        transaction_cost: float = 0.001,
        initial_capital: float = 100_000.0,
        reward_config: RewardConfig | None = None,
    ) -> None:
        super().__init__()
        if not returns.index.equals(features.index):
            raise ValueError("Returns and features must share the same date index.")

        self.returns = returns.astype(np.float32)
        self.features = features.astype(np.float32)
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        self.reward_config = reward_config or RewardConfig()

        self.asset_names = list(self.returns.columns)
        self.n_assets = len(self.asset_names)
        self.n_features = self.features.shape[1]

        observation_dim = (
            self.lookback_window * self.n_assets + self.n_features + self.n_assets + 1
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(observation_dim,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.portfolio_value = self.initial_capital
        self.return_history: list[float] = []
        self.value_history: list[float] = []
        self.turnover_history: list[float] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to the first tradable step."""

        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.portfolio_value = self.initial_capital
        self.return_history = []
        self.value_history = [self.initial_capital]
        self.turnover_history = [0.0]
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Construct the 265-dimensional state vector."""

        return_window = self.returns.iloc[
            self.current_step - self.lookback_window : self.current_step
        ].to_numpy(dtype=np.float32)
        flattened_returns = return_window.reshape(-1)
        latest_features = self.features.iloc[self.current_step].to_numpy(dtype=np.float32)
        normalized_value = np.array(
            [self.portfolio_value / self.initial_capital], dtype=np.float32
        )
        observation = np.concatenate(
            [flattened_returns, latest_features, self.weights, normalized_value]
        )
        return np.clip(observation, -10.0, 10.0).astype(np.float32)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply a single trading decision and move forward one day."""

        clipped_action = np.clip(action.astype(np.float32), 0.0, 1.0)
        if float(clipped_action.sum()) <= 1e-8:
            clipped_action = np.ones(self.n_assets, dtype=np.float32)
        new_weights = clipped_action / clipped_action.sum()

        turnover = float(np.abs(new_weights - self.weights).sum())
        transaction_penalty = turnover * self.transaction_cost
        transaction_cost_value = self.portfolio_value * transaction_penalty

        daily_asset_returns = self.returns.iloc[self.current_step].to_numpy(dtype=np.float32)
        portfolio_log_return = float(np.dot(new_weights, daily_asset_returns))

        previous_value = self.portfolio_value
        # Portfolio value evolves multiplicatively with returns but loses a
        # linear amount to trading cost because brokerage friction is paid in
        # currency units, not in abstract percentages.
        self.portfolio_value = previous_value * float(np.exp(portfolio_log_return))
        self.portfolio_value -= transaction_cost_value
        self.portfolio_value = max(self.portfolio_value, 1.0)

        realized_return = float((self.portfolio_value - previous_value) / previous_value)
        self.weights = new_weights
        self.return_history.append(realized_return)
        self.value_history.append(self.portfolio_value)
        self.turnover_history.append(turnover)

        reward = cvar_adjusted_sharpe_reward(
            return_history=self.return_history,
            weights=self.weights,
            turnover=turnover,
            config=self.reward_config,
        )

        self.current_step += 1
        terminated = self.current_step >= len(self.returns)
        truncated = False

        info = {
            "portfolio_value": self.portfolio_value,
            "weights": self.weights.copy(),
            "turnover": turnover,
            "transaction_cost_value": transaction_cost_value,
            "daily_return": realized_return,
        }

        if terminated:
            observation = self.observation_space.low.astype(np.float32)
        else:
            observation = self._get_observation()

        return observation, reward, terminated, truncated, info
