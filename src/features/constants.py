"""Feature column names — importable without triggering data/model imports."""

FEATURE_COLS = [
    # Returns (6)
    "log_return", "rolling_return_5d", "rolling_return_20d",
    "price_to_sma20", "high_low_range", "overnight_gap",
    # Volatility (5)
    "realized_vol_10d", "realized_vol_30d", "parkinson_vol", "vol_of_vol", "garch_vol",
    # Technical (5)
    "rsi_14", "macd_histogram", "bb_pct_b", "atr_14", "stochastic_k",
    # Statistical (8)
    "autocorr_lag1", "autocorr_lag5", "skewness_20d", "kurtosis_20d",
    "hurst_exponent", "sample_entropy", "rolling_sharpe_20d", "z_score_return",
]
