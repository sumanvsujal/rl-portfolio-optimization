"""Classical and sequence-model baselines for portfolio comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import pick_device, set_global_seed


def _softmax_weights(scores: np.ndarray) -> np.ndarray:
    """Project arbitrary scores to long-only portfolio weights."""

    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / (exp_scores.sum() + 1e-8)


def run_equal_weight(
    returns: pd.DataFrame,
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.001,
    rebalance_frequency: int = 21,
) -> pd.Series:
    """Run the equal-weight benchmark.

    Equal weight is deliberately simple and therefore useful: if an RL agent
    cannot beat a naive diversified benchmark, then the complexity is not yet
    justified.
    """

    n_assets = returns.shape[1]
    target_weights = np.ones(n_assets, dtype=float) / n_assets
    current_weights = target_weights.copy()
    portfolio_value = initial_capital
    history = [portfolio_value]

    for step_index, (_, row) in enumerate(returns.iterrows()):
        if step_index > 0 and step_index % rebalance_frequency == 0:
            turnover = np.abs(target_weights - current_weights).sum()
            portfolio_value -= portfolio_value * transaction_cost * turnover
            current_weights = target_weights.copy()

        portfolio_value *= float(np.exp(np.dot(current_weights, row.to_numpy(dtype=float))))
        history.append(portfolio_value)

    return pd.Series(history, index=range(len(history)), name="Equal Weight")


def _solve_mean_variance_weights(
    history: pd.DataFrame,
) -> np.ndarray:
    """Solve a long-only Sharpe-style allocation on a rolling history."""

    n_assets = history.shape[1]
    mu = history.mean().to_numpy(dtype=float)
    cov = history.cov().to_numpy(dtype=float) + np.eye(n_assets) * 1e-6

    def objective(weights: np.ndarray) -> float:
        portfolio_return = float(np.dot(weights, mu))
        portfolio_volatility = float(np.sqrt(weights @ cov @ weights))
        if portfolio_volatility <= 1e-8:
            return 0.0
        # Maximizing Sharpe is a natural baseline because the final RL reward
        # is also risk-aware. This keeps the comparison economically aligned.
        return -(portfolio_return / portfolio_volatility)

    bounds = [(0.0, 1.0)] * n_assets
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    initial_guess = np.ones(n_assets, dtype=float) / n_assets
    result = minimize(
        objective,
        x0=initial_guess,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
    )
    if not result.success:
        return initial_guess
    return np.asarray(result.x, dtype=float)


def run_mean_variance(
    returns: pd.DataFrame,
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.001,
    lookback_window: int = 60,
    rebalance_frequency: int = 21,
) -> pd.Series:
    """Run a rolling mean-variance optimization baseline."""

    n_assets = returns.shape[1]
    current_weights = np.ones(n_assets, dtype=float) / n_assets
    portfolio_value = initial_capital
    history = [portfolio_value]

    for step_index, (_, row) in enumerate(returns.iterrows()):
        if step_index >= lookback_window and step_index % rebalance_frequency == 0:
            rolling_history = returns.iloc[step_index - lookback_window : step_index]
            new_weights = _solve_mean_variance_weights(rolling_history)
            turnover = np.abs(new_weights - current_weights).sum()
            portfolio_value -= portfolio_value * transaction_cost * turnover
            current_weights = new_weights

        portfolio_value *= float(np.exp(np.dot(current_weights, row.to_numpy(dtype=float))))
        history.append(portfolio_value)

    return pd.Series(history, index=range(len(history)), name="MVO")


class AttentionLSTMForecaster(nn.Module):
    """Two-layer attention LSTM that predicts next-day asset returns."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Predict next-day return scores from a return sequence."""

        outputs, _ = self.lstm(sequence)
        final_state = outputs[:, -1, :]
        attention_scores = torch.bmm(
            self.query(final_state).unsqueeze(1),
            outputs.transpose(1, 2),
        ).squeeze(1)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), outputs).squeeze(1)
        combined = torch.cat([final_state, self.value_projection(context)], dim=-1)
        return self.output(combined)


@dataclass
class LSTMTrainingResult:
    """Bundle the trained model and its loss curve."""

    model: AttentionLSTMForecaster
    loss_history: list[float]
    mean: np.ndarray
    std: np.ndarray


def _build_sequences(data: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn a return matrix into supervised LSTM training samples."""

    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for end_index in range(sequence_length, len(data)):
        sequences.append(data[end_index - sequence_length : end_index])
        targets.append(data[end_index])
    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def train_lstm_model(
    train_returns: pd.DataFrame,
    sequence_length: int = 60,
    hidden_size: int = 256,
    num_layers: int = 2,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    device: str | None = None,
) -> LSTMTrainingResult:
    """Train the LSTM return-forecast baseline.

    The LSTM is intentionally predictive rather than directly optimizing the
    RL reward. That makes it a useful contrast: it asks whether forecasting
    next-day returns alone is enough to build a competitive portfolio policy.
    """

    set_global_seed(seed)
    device = device or pick_device()
    values = train_returns.to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0) + 1e-6
    normalized = (values - mean) / std

    features, targets = _build_sequences(normalized, sequence_length=sequence_length)
    dataset = TensorDataset(torch.tensor(features), torch.tensor(targets))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AttentionLSTMForecaster(
        input_size=train_returns.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    loss_history: list[float] = []
    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_x)
            sample_count += len(batch_x)
        loss_history.append(running_loss / max(sample_count, 1))

    return LSTMTrainingResult(model=model, loss_history=loss_history, mean=mean, std=std)


def run_lstm_strategy(
    train_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.001,
    sequence_length: int = 60,
    hidden_size: int = 256,
    num_layers: int = 2,
    epochs: int = 50,
    seed: int = 42,
    device: str | None = None,
) -> tuple[pd.Series, list[float]]:
    """Train the LSTM baseline and turn its forecasts into portfolio weights."""

    training_result = train_lstm_model(
        train_returns=train_returns,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        epochs=epochs,
        seed=seed,
        device=device,
    )

    device = device or pick_device()
    model = training_result.model.to(device)
    model.eval()

    combined_returns = pd.concat([train_returns, test_returns], axis=0)
    portfolio_value = initial_capital
    current_weights = np.ones(test_returns.shape[1], dtype=float) / test_returns.shape[1]
    history = [portfolio_value]

    for date in test_returns.index:
        end_position = combined_returns.index.get_loc(date)
        window = combined_returns.iloc[end_position - sequence_length : end_position]
        standardized_window = (window.to_numpy(dtype=np.float32) - training_result.mean) / training_result.std
        sequence_tensor = torch.tensor(standardized_window).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_scores = model(sequence_tensor).cpu().numpy().reshape(-1)

        new_weights = _softmax_weights(predicted_scores)
        turnover = np.abs(new_weights - current_weights).sum()
        portfolio_value -= portfolio_value * transaction_cost * turnover
        current_weights = new_weights

        realized_returns = test_returns.loc[date].to_numpy(dtype=float)
        portfolio_value *= float(np.exp(np.dot(current_weights, realized_returns)))
        history.append(portfolio_value)

    return pd.Series(history, index=range(len(history)), name="LSTM"), training_result.loss_history
