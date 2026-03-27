"""Return-based features: log returns, rolling returns, price ratios."""
import pandas as pd
import numpy as np


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Log return
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # 2-3. Rolling returns
    df["rolling_return_5d"] = df["close"].pct_change(5)
    df["rolling_return_20d"] = df["close"].pct_change(20)

    # 4. Price-to-SMA ratio (mean reversion signal)
    df["price_to_sma20"] = df["close"] / df["close"].rolling(20).mean()

    # 5. Intraday range (volatility proxy)
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]

    # 6. Overnight gap
    df["overnight_gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    return df
