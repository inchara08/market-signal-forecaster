"""Technical indicator features: RSI, MACD, Bollinger, ATR, Stochastic."""
import pandas as pd
import numpy as np


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Use the 'ta' library if available, else fall back to manual implementations
    try:
        import ta
        _add_with_ta(df)
    except ImportError:
        _add_manual(df)

    return df


def _add_with_ta(df: pd.DataFrame) -> None:
    import ta

    # 1. RSI (14-period)
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # 2. MACD histogram
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd_histogram"] = macd.macd_diff()

    # 3. Bollinger Band %B
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_pct_b"] = bb.bollinger_pband()

    # 4. Average True Range (normalized)
    df["atr_14"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range() / df["close"]

    # 5. Stochastic %K
    stoch = ta.momentum.StochasticOscillator(
        df["high"], df["low"], df["close"], window=14, smooth_window=3
    )
    df["stochastic_k"] = stoch.stoch()


def _add_manual(df: pd.DataFrame) -> None:
    """Fallback manual implementations."""
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD histogram
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = macd_line - signal_line

    # Bollinger %B
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df["bb_pct_b"] = (df["close"] - lower) / (upper - lower)

    # ATR normalized
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean() / df["close"]

    # Stochastic %K
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    df["stochastic_k"] = 100 * (df["close"] - low14) / (high14 - low14)
