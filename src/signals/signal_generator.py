"""Generate buy/sell/hold signals from model predictions."""
import numpy as np
import pandas as pd


def generate_signals(
    predictions: pd.Series,
    buy_threshold: float = 0.005,
    sell_threshold: float = -0.005,
) -> pd.DataFrame:
    """
    Convert predicted returns to discrete signals.

    Returns a DataFrame with:
    - signal: 1 (buy), -1 (sell), 0 (hold)
    - confidence: abs(predicted_return) normalized to [0, 1]
    """
    signals = pd.Series(0, index=predictions.index, dtype=int)
    signals[predictions > buy_threshold] = 1
    signals[predictions < sell_threshold] = -1

    max_pred = predictions.abs().max()
    confidence = (predictions.abs() / max_pred).clip(0, 1) if max_pred > 0 else predictions.abs()

    return pd.DataFrame({
        "predicted_return": predictions,
        "signal": signals,
        "confidence": confidence,
    })


def backtest_signals(
    signals_df: pd.DataFrame,
    actual_returns: pd.Series,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """
    Simulate a long-only regime strategy:
      - signal=+1 (low-vol regime): invested, earns daily return
      - signal=-1 or 0 (high-vol regime): in cash, earns 0
    Transaction cost applied on regime switches.
    """
    df = signals_df.copy()
    df["actual_return"] = actual_returns

    # Long-only: +1 = invested, anything else = cash (0)
    position = df["signal"].shift(1).clip(lower=0)   # 1 when long, 0 when in cash
    df["signal_change"] = df["signal"].diff().abs()
    df["strategy_return"] = position * df["actual_return"] - df["signal_change"] * transaction_cost

    df["cumulative_strategy"] = (1 + df["strategy_return"].fillna(0)).cumprod()
    df["cumulative_buyhold"] = (1 + df["actual_return"].fillna(0)).cumprod()

    return df
