"""Volatility features: realized vol, Parkinson, vol-of-vol, GARCH."""
import pandas as pd
import numpy as np


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    log_ret = np.log(df["close"] / df["close"].shift(1))

    # 1. Realized volatility 10-day (annualized)
    df["realized_vol_10d"] = log_ret.rolling(10).std() * np.sqrt(252)

    # 2. Realized volatility 30-day (annualized)
    df["realized_vol_30d"] = log_ret.rolling(30).std() * np.sqrt(252)

    # 3. Parkinson volatility (range-based, lower variance than close-to-close)
    log_hl = np.log(df["high"] / df["low"])
    df["parkinson_vol"] = np.sqrt(
        (1 / (4 * np.log(2))) * log_hl.rolling(20).apply(lambda x: (x**2).mean(), raw=True)
    ) * np.sqrt(252)

    # 4. Volatility-of-volatility
    df["vol_of_vol"] = df["realized_vol_10d"].rolling(20).std()

    # 5. GARCH(1,1) conditional volatility
    df["garch_vol"] = _rolling_garch(log_ret, window=252)

    return df


def _rolling_garch(returns: pd.Series, window: int = 252) -> pd.Series:
    """Compute rolling GARCH(1,1) one-step-ahead conditional std."""
    try:
        from arch import arch_model
    except ImportError:
        return pd.Series(np.nan, index=returns.index)

    result = pd.Series(np.nan, index=returns.index)
    r = returns.dropna() * 100  # scale to % for numerical stability

    for i in range(window, len(r)):
        window_data = r.iloc[i - window: i]
        try:
            am = arch_model(window_data, vol="Garch", p=1, q=1, rescale=False)
            fit = am.fit(disp="off", show_warning=False)
            forecast = fit.forecast(horizon=1)
            cond_var = forecast.variance.iloc[-1, 0]
            result.iloc[r.index.get_loc(r.index[i])] = np.sqrt(cond_var) / 100 * np.sqrt(252)
        except Exception:
            pass

    return result
