"""Reward utilities for the CVaR-adjusted Sharpe objective."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    """Configuration for the CVaR-adjusted Sharpe reward."""

    rolling_window: int = 60
    cvar_alpha: float = 0.05
    lambda_cvar: float = 2.0
    gamma_hhi: float = 0.02
    delta_transaction_cost: float = 5.0


def herfindahl_hirschman_index(weights: np.ndarray) -> float:
    """Measure portfolio concentration through the HHI."""

    return float(np.sum(np.square(weights)))


def conditional_value_at_risk(
    returns: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Compute CVaR on the worst alpha fraction of returns.

    We use CVaR instead of plain VaR because it punishes the severity of the
    left tail, not just the cutoff. In portfolio management, the magnitude of
    the bad days often matters more than the existence of a threshold.
    """

    if returns.size == 0:
        return 0.0
    sorted_returns = np.sort(returns)
    cutoff = max(1, int(np.ceil(sorted_returns.size * alpha)))
    tail = sorted_returns[:cutoff]
    return float(np.mean(tail))


def rolling_sharpe_ratio(returns: np.ndarray) -> float:
    """Compute an annualized Sharpe ratio for a return window."""

    if returns.size == 0:
        return 0.0
    sigma = float(np.std(returns))
    if sigma < 1e-8:
        return 0.0
    return float((np.mean(returns) / sigma) * np.sqrt(252.0))


def cvar_adjusted_sharpe_reward(
    return_history: list[float],
    weights: np.ndarray,
    turnover: float,
    config: RewardConfig | None = None,
) -> float:
    """Compute the project's final reward function.

    The reward intentionally combines four forces:

    - Sharpe rewards persistent risk-adjusted performance.
    - CVaR penalizes tail events that volatility alone can miss.
    - HHI discourages fragile one-asset concentration.
    - Transaction-cost penalty discourages unrealistic overtrading.
    """

    config = config or RewardConfig()
    window_returns = np.asarray(return_history[-config.rolling_window :], dtype=float)
    if window_returns.size < config.rolling_window:
        # Early in an episode, the agent does not yet have enough history for
        # a reliable 60-day risk estimate, so raw return is the least biased
        # signal available.
        return float(window_returns[-1] if window_returns.size else 0.0)

    sharpe_component = rolling_sharpe_ratio(window_returns)
    cvar_component = abs(conditional_value_at_risk(window_returns, alpha=config.cvar_alpha))
    concentration_component = herfindahl_hirschman_index(weights)
    reward = (
        sharpe_component
        - config.lambda_cvar * cvar_component
        - config.gamma_hhi * concentration_component
        - config.delta_transaction_cost * turnover
    )
    return float(reward)
