"""Verify feature values against known ground truth."""
import numpy as np
import pandas as pd
import pytest

from src.features.returns import add_return_features
from src.features.technical import add_technical_features
from src.features.volatility import add_volatility_features
from src.features.statistical import add_statistical_features


@pytest.fixture
def sample_df():
    """Generate 200 rows of synthetic OHLCV data."""
    np.random.seed(42)
    n = 200
    close = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n))
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "open": close * np.random.uniform(0.995, 1.005, n),
        "high": close * np.random.uniform(1.001, 1.015, n),
        "low": close * np.random.uniform(0.985, 0.999, n),
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)
    return df


class TestReturnFeatures:
    def test_log_return_shape(self, sample_df):
        df = add_return_features(sample_df)
        assert "log_return" in df.columns
        assert len(df) == len(sample_df)

    def test_log_return_values(self, sample_df):
        df = add_return_features(sample_df)
        expected = np.log(sample_df["close"] / sample_df["close"].shift(1))
        pd.testing.assert_series_equal(df["log_return"], expected, check_names=False)

    def test_rolling_returns_lag(self, sample_df):
        df = add_return_features(sample_df)
        expected_5d = sample_df["close"].pct_change(5)
        pd.testing.assert_series_equal(df["rolling_return_5d"], expected_5d, check_names=False)

    def test_high_low_range_positive(self, sample_df):
        df = add_return_features(sample_df)
        assert (df["high_low_range"].dropna() >= 0).all()

    def test_all_return_features_present(self, sample_df):
        df = add_return_features(sample_df)
        expected = ["log_return", "rolling_return_5d", "rolling_return_20d",
                    "price_to_sma20", "high_low_range", "overnight_gap"]
        for col in expected:
            assert col in df.columns, f"Missing feature: {col}"


class TestTechnicalFeatures:
    def test_rsi_bounds(self, sample_df):
        df = add_technical_features(sample_df)
        rsi = df["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI must be in [0, 100]"

    def test_stochastic_bounds(self, sample_df):
        df = add_technical_features(sample_df)
        stoch = df["stochastic_k"].dropna()
        assert (stoch >= 0).all() and (stoch <= 100).all(), "Stochastic must be in [0, 100]"

    def test_bb_pct_b_values(self, sample_df):
        df = add_technical_features(sample_df)
        assert "bb_pct_b" in df.columns

    def test_atr_positive(self, sample_df):
        df = add_technical_features(sample_df)
        assert (df["atr_14"].dropna() >= 0).all()

    def test_all_technical_features_present(self, sample_df):
        df = add_technical_features(sample_df)
        for col in ["rsi_14", "macd_histogram", "bb_pct_b", "atr_14", "stochastic_k"]:
            assert col in df.columns, f"Missing feature: {col}"


class TestVolatilityFeatures:
    def test_realized_vol_positive(self, sample_df):
        df = add_volatility_features(sample_df)
        assert (df["realized_vol_10d"].dropna() >= 0).all()
        assert (df["realized_vol_30d"].dropna() >= 0).all()

    def test_parkinson_vol_positive(self, sample_df):
        df = add_volatility_features(sample_df)
        assert (df["parkinson_vol"].dropna() >= 0).all()

    def test_all_volatility_features_present(self, sample_df):
        df = add_volatility_features(sample_df)
        for col in ["realized_vol_10d", "realized_vol_30d", "parkinson_vol", "vol_of_vol"]:
            assert col in df.columns, f"Missing feature: {col}"


class TestStatisticalFeatures:
    def test_skewness_finite(self, sample_df):
        df = add_statistical_features(sample_df)
        skew = df["skewness_20d"].dropna()
        assert np.isfinite(skew).all()

    def test_rolling_sharpe_present(self, sample_df):
        df = add_statistical_features(sample_df)
        assert "rolling_sharpe_20d" in df.columns

    def test_feature_count(self, sample_df):
        from src.features.constants import FEATURE_COLS
        df = sample_df.copy()
        df = add_return_features(df)
        df = add_volatility_features(df)
        df = add_technical_features(df)
        df = add_statistical_features(df)
        present = [c for c in FEATURE_COLS if c in df.columns]
        assert len(present) >= 20, f"Expected 20+ features, got {len(present)}: {present}"
