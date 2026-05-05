"""Training entry point for PPO, A2C, and SAC."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .environment import PortfolioEnv
from .features import build_feature_matrix
from .reward import RewardConfig
from .utils import (
    ensure_directory,
    pick_device,
    preprocess_market_data,
    set_global_seed,
    split_train_test,
)


def _make_vec_env(train_returns, train_features, transaction_cost: float, seed: int) -> DummyVecEnv:
    """Construct the monitored training environment."""

    def factory() -> Monitor:
        env = PortfolioEnv(
            returns=train_returns,
            features=train_features,
            transaction_cost=transaction_cost,
            reward_config=RewardConfig(),
        )
        env.reset(seed=seed)
        return Monitor(env)

    return DummyVecEnv([factory])


def build_agent(
    algorithm: str,
    env: DummyVecEnv,
    device: str,
    seed: int,
) -> PPO | A2C | SAC:
    """Build a Stable-Baselines3 agent with research-aligned defaults."""

    algorithm = algorithm.lower()
    if algorithm == "ppo":
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=2e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            seed=seed,
            verbose=1,
            device=device,
            policy_kwargs={"net_arch": [256, 256, 128]},
        )
    if algorithm == "a2c":
        return A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=seed,
            verbose=1,
            device=device,
            policy_kwargs={"net_arch": [256, 256, 128]},
        )
    if algorithm == "sac":
        return SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=200_000,
            learning_starts=1_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            seed=seed,
            verbose=1,
            device=device,
            policy_kwargs={"net_arch": [256, 256, 128]},
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def train_agents(
    algorithms: list[str],
    timesteps: int,
    seed: int,
    models_dir: Path,
    transaction_cost: float = 0.001,
) -> None:
    """Train one or more RL agents on the fixed train split."""

    set_global_seed(seed)
    prices, returns = preprocess_market_data()
    features = build_feature_matrix(prices=prices, returns=returns)
    _, train_returns, _, _ = split_train_test(prices=prices, returns=returns)
    train_features = features.loc[train_returns.index].copy()

    models_dir = ensure_directory(models_dir)
    device = pick_device()

    for algorithm in algorithms:
        print(f"\nTraining {algorithm.upper()} on 2010-2021 data")
        env = _make_vec_env(
            train_returns=train_returns,
            train_features=train_features,
            transaction_cost=transaction_cost,
            seed=seed,
        )
        model = build_agent(algorithm=algorithm, env=env, device=device, seed=seed)
        # We keep the long training horizon because financial environments are
        # noisy; shorter runs tend to produce unstable conclusion.
        model.learn(total_timesteps=timesteps, progress_bar=True)
        output_path = models_dir / f"{algorithm.lower()}_portfolio_model"
        model.save(str(output_path))
        env.close()
        print(f"Saved {algorithm.upper()} model to {output_path}.zip")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["ppo", "a2c", "sac"],
        choices=["ppo", "a2c", "sac"],
        help="RL algorithms to train.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=600_000,
        help="Training timesteps per RL agent.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible experiments.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory where trained models will be saved.",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="Transaction cost per unit turnover.",
    )
    return parser.parse_args()


def main() -> None:
    """Train the selected RL models."""

    args = parse_args()
    train_agents(
        algorithms=args.algorithms,
        timesteps=args.timesteps,
        seed=args.seed,
        models_dir=args.models_dir,
        transaction_cost=args.transaction_cost,
    )


if __name__ == "__main__":
    main()
