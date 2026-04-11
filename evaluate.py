"""Evaluation, metrics, and figure generation for the final benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines3 import A2C, PPO, SAC
from pandas.tseries.offsets import BDay

from .baselines import run_equal_weight, run_lstm_strategy, run_mean_variance
from .environment import PortfolioEnv
from .features import build_feature_matrix
from .utils import (
    ensure_directory,
    preprocess_market_data,
    set_global_seed,
    split_train_test,
)


def _history_index(test_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Create a portfolio-history index including the day before the test set."""

    initial_day = test_index[0] - BDay(1)
    return pd.DatetimeIndex([initial_day, *test_index.to_pydatetime().tolist()])


def evaluate_rl_model(
    model: PPO | A2C | SAC,
    test_returns: pd.DataFrame,
    test_features: pd.DataFrame,
    transaction_cost: float = 0.001,
    initial_capital: float = 100_000.0,
    label: str = "RL",
) -> pd.Series:
    """Run a trained RL model on the test period."""

    env = PortfolioEnv(
        returns=test_returns,
        features=test_features,
        transaction_cost=transaction_cost,
        initial_capital=initial_capital,
    )
    observation, _ = env.reset()
    done = False
    values = [initial_capital]

    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, done, _, info = env.step(action)
        values.append(info["portfolio_value"])

    return pd.Series(values, index=_history_index(test_returns.index), name=label)


def calculate_metrics(portfolio_values: pd.Series) -> dict[str, float]:
    """Compute the standard portfolio-performance metrics."""

    daily_returns = portfolio_values.pct_change().dropna()
    cumulative_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1.0
    annual_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (
        252.0 / max(len(daily_returns), 1)
    ) - 1.0
    annual_volatility = float(daily_returns.std() * np.sqrt(252.0))
    sharpe_ratio = (
        float(daily_returns.mean() / (daily_returns.std() + 1e-8) * np.sqrt(252.0))
        if not daily_returns.empty
        else 0.0
    )
    downside = daily_returns[daily_returns < 0.0]
    sortino_ratio = (
        float(daily_returns.mean() / (downside.std() + 1e-8) * np.sqrt(252.0))
        if not daily_returns.empty
        else 0.0
    )
    running_peak = portfolio_values.cummax()
    drawdowns = (portfolio_values / running_peak) - 1.0
    max_drawdown = float(drawdowns.min())
    calmar_ratio = float(annual_return / (abs(max_drawdown) + 1e-8))
    var_95 = float(np.percentile(daily_returns, 5)) if not daily_returns.empty else 0.0
    cvar_tail = daily_returns[daily_returns <= var_95]
    cvar_95 = float(cvar_tail.mean()) if not cvar_tail.empty else var_95

    return {
        "Cumulative Return (%)": cumulative_return * 100.0,
        "Annual Return (%)": annual_return * 100.0,
        "Annual Volatility (%)": annual_volatility * 100.0,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown (%)": max_drawdown * 100.0,
        "Calmar Ratio": calmar_ratio,
        "Hit Ratio (%)": float((daily_returns > 0.0).mean() * 100.0),
        "VaR 95% (%)": var_95 * 100.0,
        "CVaR 95% (%)": cvar_95 * 100.0,
        "Final Value": float(portfolio_values.iloc[-1]),
    }


def plot_data_quality(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a compact data-quality figure for the README and report."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    normalized_prices = prices / prices.iloc[0]
    normalized_prices.plot(ax=axes[0, 0], linewidth=1.2)
    axes[0, 0].set_title("Normalized ETF Prices")
    axes[0, 0].set_ylabel("Growth of 1.0")
    axes[0, 0].grid(True, alpha=0.3)

    sns.heatmap(returns.corr(), cmap="RdYlGn", center=0.0, ax=axes[0, 1])
    axes[0, 1].set_title("Return Correlation")

    rolling_volatility = returns.rolling(30).std() * np.sqrt(252.0)
    rolling_volatility.plot(ax=axes[1, 0], linewidth=1.2)
    axes[1, 0].set_title("30-Day Rolling Volatility")
    axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    axes[1, 0].grid(True, alpha=0.3)

    annual_returns = returns.resample("YE").sum()
    annual_returns.index = annual_returns.index.year
    annual_returns.mean().sort_values().plot(kind="barh", ax=axes[1, 1], color="#2E8B57")
    axes[1, 1].set_title("Average Annual Log Return")
    axes[1, 1].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_paper_results(
    portfolio_curves: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create the main comparison figure used in the README."""

    colors = {
        "Equal Weight": "#757575",
        "MVO": "#1F77B4",
        "LSTM": "#8C564B",
        "A2C": "#FF7F0E",
        "SAC": "#2CA02C",
        "PPO": "#D62728",
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    growth = portfolio_curves / portfolio_curves.iloc[0]
    for column in growth.columns:
        axes[0, 0].plot(growth.index, growth[column], label=column, color=colors.get(column), linewidth=2.0)
    axes[0, 0].set_title("Out-of-Sample Portfolio Growth (2022-2024)")
    axes[0, 0].set_ylabel("Growth Multiple")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    for column in portfolio_curves.columns:
        running_peak = portfolio_curves[column].cummax()
        drawdown = (portfolio_curves[column] / running_peak) - 1.0
        axes[0, 1].plot(drawdown.index, drawdown * 100.0, label=column, color=colors.get(column), linewidth=2.0)
    axes[0, 1].set_title("Drawdown Profiles")
    axes[0, 1].set_ylabel("Drawdown (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    for strategy, row in metrics_frame.iterrows():
        axes[1, 0].scatter(
            row["Annual Volatility (%)"],
            row["Annual Return (%)"],
            s=180,
            color=colors.get(strategy),
            label=strategy,
        )
        axes[1, 0].annotate(strategy, (row["Annual Volatility (%)"], row["Annual Return (%)"]), xytext=(6, 4), textcoords="offset points")
    axes[1, 0].set_title("Risk-Return Tradeoff")
    axes[1, 0].set_xlabel("Annual Volatility (%)")
    axes[1, 0].set_ylabel("Annual Return (%)")
    axes[1, 0].grid(True, alpha=0.3)

    metrics_frame["Sharpe Ratio"].plot(kind="bar", ax=axes[1, 1], color=[colors.get(name) for name in metrics_frame.index])
    axes[1, 1].set_title("Sharpe Ratio Comparison")
    axes[1, 1].set_ylabel("Sharpe Ratio")
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    axes[1, 1].tick_params(axis="x", rotation=25)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lstm_training_curve(loss_history: list[float], output_path: Path) -> None:
    """Plot the LSTM training loss history."""

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_history, color="#8C564B", linewidth=2.0)
    ax.set_title("LSTM Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_model(algorithm: str, models_dir: Path) -> PPO | A2C | SAC:
    """Load a trained RL model from disk."""

    path = models_dir / f"{algorithm}_portfolio_model.zip"
    if algorithm == "ppo":
        return PPO.load(str(path))
    if algorithm == "a2c":
        return A2C.load(str(path))
    if algorithm == "sac":
        return SAC.load(str(path))
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def evaluate_pipeline(
    models_dir: Path,
    results_dir: Path,
    seed: int,
    transaction_cost: float = 0.001,
) -> None:
    """Run the full benchmark evaluation pipeline."""

    set_global_seed(seed)
    figures_dir = ensure_directory(results_dir / "figures")

    prices, returns = preprocess_market_data()
    features = build_feature_matrix(prices=prices, returns=returns)
    train_prices, train_returns, test_prices, test_returns = split_train_test(
        prices=prices,
        returns=returns,
    )
    test_features = features.loc[test_returns.index].copy()

    portfolio_curves = pd.DataFrame(index=_history_index(test_returns.index))

    for algorithm, label in [("ppo", "PPO"), ("a2c", "A2C"), ("sac", "SAC")]:
        model = _load_model(algorithm=algorithm, models_dir=models_dir)
        portfolio_curves[label] = evaluate_rl_model(
            model=model,
            test_returns=test_returns,
            test_features=test_features,
            transaction_cost=transaction_cost,
            label=label,
        )

    equal_weight_curve = run_equal_weight(
        returns=test_returns,
        transaction_cost=transaction_cost,
    )
    portfolio_curves["Equal Weight"] = equal_weight_curve.to_numpy()

    mvo_curve = run_mean_variance(
        returns=test_returns,
        transaction_cost=transaction_cost,
    )
    portfolio_curves["MVO"] = mvo_curve.to_numpy()

    lstm_curve, loss_history = run_lstm_strategy(
        train_returns=train_returns,
        test_returns=test_returns,
        transaction_cost=transaction_cost,
        seed=seed,
    )
    portfolio_curves["LSTM"] = lstm_curve.to_numpy()

    metrics = {
        strategy: calculate_metrics(portfolio_curves[strategy])
        for strategy in portfolio_curves.columns
    }
    metrics_frame = pd.DataFrame(metrics).T
    metrics_frame = metrics_frame.loc[["PPO", "SAC", "A2C", "MVO", "Equal Weight", "LSTM"]]

    metrics_frame.to_csv(results_dir / "metrics.csv")
    portfolio_curves.to_csv(results_dir / "returns.csv", index_label="Date")

    plot_data_quality(prices=prices, returns=returns, output_path=figures_dir / "data_quality.png")
    plot_paper_results(
        portfolio_curves=portfolio_curves,
        metrics_frame=metrics_frame,
        output_path=figures_dir / "paper_results.png",
    )
    plot_lstm_training_curve(
        loss_history=loss_history,
        output_path=figures_dir / "lstm_training.png",
    )

    print(f"Saved metrics to {results_dir / 'metrics.csv'}")
    print(f"Saved return paths to {results_dir / 'returns.csv'}")
    print(f"Saved figures to {figures_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing trained PPO, A2C, and SAC models.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory where metrics and figures will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the LSTM baseline and deterministic setup.",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="Transaction cost per unit turnover.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the evaluation entry point."""

    args = parse_args()
    evaluate_pipeline(
        models_dir=args.models_dir,
        results_dir=ensure_directory(args.results_dir),
        seed=args.seed,
        transaction_cost=args.transaction_cost,
    )


if __name__ == "__main__":
    main()
