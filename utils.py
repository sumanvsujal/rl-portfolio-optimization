"""Utility helpers for data access, preprocessing, and reproducibility."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yfinance as yf


DEFAULT_TICKERS: list[str] = [
    "SPY",
    "QQQ",
    "TLT",
    "GLD",
    "SLV",
    "NIFTYBEES.NS",
    "BANKBEES.NS",
    "JUNIORBEES.NS",
    "GOLDBEES.NS",
]

TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_global_seed(seed: int) -> None:
    """Seed the major random number generators used in the project.

    Reproducibility matters more in RL than in many supervised tasks because
    a small change in initialization can propagate through the whole training
    trajectory. Setting the seed does not make finance deterministic, but it
    does make experiments easier to compare honestly.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_price_data(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """Download adjusted close prices for the fixed 9-ETF universe.

    The project uses live downloads rather than shipping raw market CSVs
    because that keeps the repository lightweight and makes the full
    experimental pipeline reproducible from code.
    """

    tickers = list(tickers)
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    prices = raw["Close"].copy()
    prices = prices.reindex(columns=tickers)
    prices = prices.sort_index().ffill().bfill()
    missing_columns = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing_columns:
        raise ValueError(f"Missing tickers after download: {missing_columns}")
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert prices into daily log returns.

    Log returns are used for both USD and INR assets so the modeling
    convention stays consistent across markets and through time.
    """

    returns = np.log(prices / prices.shift(1)).dropna()
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def preprocess_market_data(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and preprocess the market panel."""

    prices = download_price_data(tickers=tickers, start=start, end=end)
    returns = compute_log_returns(prices)
    prices = prices.loc[returns.index]
    return prices, returns


def split_train_test(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into the fixed train and test windows."""

    train_prices = prices.loc[:train_end].copy()
    train_returns = returns.loc[:train_end].copy()
    test_prices = prices.loc[test_start:].copy()
    test_returns = returns.loc[test_start:].copy()
    return train_prices, train_returns, test_prices, test_returns


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def pick_device() -> str:
    """Pick the best available PyTorch device string."""

    return "cuda" if torch.cuda.is_available() else "cpu"
