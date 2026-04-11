"""Feature engineering for the fixed 265-dimensional RL state."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Compute a normalized RSI signal in the range roughly [-1, 1]."""

    delta = prices.diff()
    gains = delta.clip(lower=0).rolling(window).mean()
    losses = (-delta).clip(lower=0).rolling(window).mean()
    rs = gains / (losses + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return (rsi - 50.0) / 50.0


def _rolling_drawdown(prices: pd.Series, window: int = 60) -> pd.Series:
    """Compute drawdown from the rolling peak."""

    rolling_peak = prices.rolling(window).max()
    return (prices / (rolling_peak + 1e-8)) - 1.0


def build_feature_matrix(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Create the fixed 75-feature state representation.

    The user-facing project summary specifies a 265-dimensional state:
    180 return-history values + 75 engineered features + 9 weights + 1
    normalized portfolio value. To preserve that contract, this module
    emits exactly 75 engineered features on every trading day.

    We use 8 asset-level signals for each of the 9 ETFs plus 3 macro
    signals shared across the whole market panel:

    - RSI (14d)
    - 5-day moving-average ratio
    - 20-day moving-average ratio
    - 60-day moving-average ratio
    - 5-day momentum
    - 20-day momentum
    - 20-day volatility
    - 60-day drawdown
    - rolling average cross-asset correlation
    - equal-weight market momentum
    - cross-sectional return dispersion

    The motivation is economic rather than purely predictive: the agent
    should observe trend, reversion, risk, crowding, and market regime
    variables, not just raw returns.
    """

    features: dict[str, pd.Series] = {}

    for ticker in prices.columns:
        asset_prices = prices[ticker]
        asset_returns = returns[ticker]
        safe_name = ticker.replace(".", "_")

        ma_5 = asset_prices.rolling(5).mean()
        ma_20 = asset_prices.rolling(20).mean()
        ma_60 = asset_prices.rolling(60).mean()

        features[f"{safe_name}_rsi_14"] = _rsi(asset_prices, window=14)
        features[f"{safe_name}_ma_5_ratio"] = (asset_prices / (ma_5 + 1e-8)) - 1.0
        features[f"{safe_name}_ma_20_ratio"] = (asset_prices / (ma_20 + 1e-8)) - 1.0
        features[f"{safe_name}_ma_60_ratio"] = (asset_prices / (ma_60 + 1e-8)) - 1.0
        features[f"{safe_name}_momentum_5"] = asset_returns.rolling(5).sum()
        features[f"{safe_name}_momentum_20"] = asset_returns.rolling(20).sum()
        features[f"{safe_name}_volatility_20"] = asset_returns.rolling(20).std() * np.sqrt(252.0)
        features[f"{safe_name}_drawdown_60"] = _rolling_drawdown(asset_prices, window=60)

    corr_signal: list[float] = []
    dispersion_signal: list[float] = []
    market_momentum_signal: list[float] = []
    return_values = returns.values
    n_assets = returns.shape[1]

    for row_index in range(len(returns)):
        if row_index < 20:
            corr_signal.append(0.0)
            dispersion_signal.append(0.0)
            market_momentum_signal.append(0.0)
            continue

        window_slice = return_values[row_index - 20 : row_index]
        correlation_matrix = np.corrcoef(window_slice.T)
        upper_triangle = np.triu_indices(n_assets, k=1)
        corr_signal.append(float(np.nanmean(correlation_matrix[upper_triangle])))
        dispersion_signal.append(float(np.nanstd(return_values[row_index])))
        market_momentum_signal.append(float(window_slice.mean()))

    feature_frame = pd.DataFrame(features, index=returns.index)
    feature_frame["macro_avg_corr_20"] = corr_signal
    feature_frame["macro_market_momentum_20"] = market_momentum_signal
    feature_frame["macro_cross_sectional_dispersion"] = dispersion_signal
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.ffill().bfill().fillna(0.0)
    feature_frame = feature_frame.clip(lower=-5.0, upper=5.0)

    if feature_frame.shape[1] != 75:
        raise ValueError(
            f"Expected 75 engineered features, found {feature_frame.shape[1]}."
        )

    return feature_frame
