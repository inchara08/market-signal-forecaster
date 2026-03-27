"""Statistical features: autocorrelation, moments, Hurst, entropy, Sharpe, z-score."""
import pandas as pd
import numpy as np


def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    log_ret = np.log(df["close"] / df["close"].shift(1))

    # 1. Rolling lag-1 autocorrelation
    df["autocorr_lag1"] = log_ret.rolling(20).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )

    # 2. Rolling lag-5 autocorrelation
    df["autocorr_lag5"] = log_ret.rolling(20).apply(
        lambda x: pd.Series(x).autocorr(lag=5), raw=False
    )

    # 3. Rolling skewness
    df["skewness_20d"] = log_ret.rolling(20).skew()

    # 4. Rolling excess kurtosis
    df["kurtosis_20d"] = log_ret.rolling(20).kurt()

    # 5. Hurst exponent (trend/mean-reversion indicator)
    df["hurst_exponent"] = log_ret.rolling(100).apply(_hurst, raw=True)

    # 6. Sample entropy
    df["sample_entropy"] = log_ret.rolling(50).apply(_sample_entropy, raw=True)

    # 7. Rolling Sharpe ratio (20-day, annualized)
    df["rolling_sharpe_20d"] = (
        log_ret.rolling(20).mean() / log_ret.rolling(20).std()
    ) * np.sqrt(252)

    # 8. Z-score of return (vs 20-day rolling stats)
    df["z_score_return"] = (
        (log_ret - log_ret.rolling(20).mean()) / log_ret.rolling(20).std()
    )

    return df


def _hurst(x: np.ndarray) -> float:
    """Estimate Hurst exponent via rescaled range analysis."""
    n = len(x)
    if n < 20:
        return np.nan
    try:
        lags = range(2, min(20, n // 2))
        rs_vals = []
        for lag in lags:
            sub = x[:lag]
            mean = np.mean(sub)
            devs = np.cumsum(sub - mean)
            r = np.max(devs) - np.min(devs)
            s = np.std(sub, ddof=1)
            if s > 0:
                rs_vals.append(r / s)
        if len(rs_vals) < 2:
            return np.nan
        log_lags = np.log(list(range(2, 2 + len(rs_vals))))
        log_rs = np.log(rs_vals)
        slope, _ = np.polyfit(log_lags, log_rs, 1)
        return slope
    except Exception:
        return np.nan


def _sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Compute sample entropy of a time series window."""
    try:
        import antropy as ant
        return ant.sample_entropy(x)
    except ImportError:
        pass

    # Manual fallback
    n = len(x)
    if n < 10:
        return np.nan
    r = r_factor * np.std(x, ddof=1)
    if r == 0:
        return np.nan

    def _count_matches(template_len):
        count = 0
        for i in range(n - template_len):
            for j in range(i + 1, n - template_len):
                if np.max(np.abs(x[i:i+template_len] - x[j:j+template_len])) < r:
                    count += 1
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)
    if B == 0:
        return np.nan
    return -np.log(A / B)
