"""Clean raw OHLCV data: forward-fill gaps, cap outliers, validate OHLC logic."""
import pandas as pd
import numpy as np


def forward_fill_gaps(df: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """Forward-fill up to max_gap consecutive missing close values."""
    df = df.copy()
    df["close"] = df["close"].ffill(limit=max_gap)
    return df


def cap_outliers(df: pd.DataFrame, col: str = "close", sigma: int = 5) -> pd.DataFrame:
    """Cap values beyond ±sigma rolling standard deviations (window=30)."""
    df = df.copy()
    roll = df[col].rolling(30, min_periods=5)
    upper = roll.mean() + sigma * roll.std()
    lower = roll.mean() - sigma * roll.std()
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where OHLC logical constraints are violated."""
    mask = (
        (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
        & (df["low"] > 0)
    )
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        print(f"[cleaner] Dropping {n_dropped} rows with invalid OHLC")
    return df[mask].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = forward_fill_gaps(df)
    df = cap_outliers(df)
    df = validate_ohlc(df)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df
