"""Dash callback registrations."""
import os
import pandas as pd
from dash import Input, Output
import yaml

from src.dashboard.layouts.overview import build_price_chart
from src.dashboard.layouts.signal_view import build_signal_chart, build_pnl_chart
from src.signals.signal_generator import generate_signals, backtest_signals

# Cache regime labels per ticker so we don't re-fit HMM on every render
_regime_cache: dict = {}


def load_feature_df(ticker: str, cfg: dict) -> pd.DataFrame | None:
    path = os.path.join(cfg["data"]["processed_dir"], f"{ticker}_features.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def load_cv_results(cfg: dict) -> pd.DataFrame | None:
    path = os.path.join(cfg["data"]["processed_dir"], "cv_results.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def add_regime_labels(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Fit HMM and add volatility_regime column; cached per ticker."""
    if ticker in _regime_cache:
        df = df.copy()
        df["volatility_regime"] = _regime_cache[ticker].reindex(df.index)
        return df
    try:
        from src.models.regime import RegimeDetector
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rd = RegimeDetector(n_components=2)
            df = rd.fit_predict(df)
            _regime_cache[ticker] = df["volatility_regime"].copy()
    except Exception:
        pass
    return df


def _regime_signal(df: pd.DataFrame) -> pd.Series:
    """
    Signal directly from HMM volatility regime:
      - Low-vol regime (0)  → +1.0 (buy / stay long)
      - High-vol regime (1) → -1.0 (sell / go to cash)
    This is the core model output — not a technical indicator proxy.
    """
    if "volatility_regime" not in df.columns:
        return pd.Series(0.0, index=df.index)
    # Map: 0 → +1 (low vol = bullish), 1 → -1 (high vol = risk-off)
    return df["volatility_regime"].map({0: 1.0, 1: -1.0}).fillna(0.0)


def _compute_signal_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite signal score from engineered features.
    Range roughly [-1, +1]: positive = bullish, negative = bearish.
    Components:
      - RSI: oversold (<30) → bullish, overbought (>70) → bearish
      - MACD histogram direction
      - 5-day momentum
    """
    import numpy as np
    score = pd.Series(0.0, index=df.index)

    if "rsi_14" in df.columns:
        rsi = df["rsi_14"]
        score += ((70 - rsi) / 40).clip(-1, 1)   # +1 when RSI=30, -1 when RSI=110

    if "macd_histogram" in df.columns:
        macd = df["macd_histogram"]
        score += (macd / macd.abs().rolling(60).mean().replace(0, 1)).clip(-1, 1)

    if "rolling_return_5d" in df.columns:
        mom = df["rolling_return_5d"]
        score += (mom / mom.abs().rolling(60).mean().replace(0, 1)).clip(-1, 1)

    # Normalise to [-1, 1]
    roll_std = score.rolling(60).std().replace(0, 1)
    score = (score / (3 * roll_std)).clip(-1, 1)
    return score.fillna(0)


def register_callbacks(app, cfg: dict, cache) -> None:

    @app.callback(
        Output("overview-chart", "figure"),
        Input("overview-ticker", "value"),
        Input("overview-daterange", "start_date"),
        Input("overview-daterange", "end_date"),
    )
    def update_overview(ticker, start_date, end_date):
        df = load_feature_df(ticker, cfg)
        if df is None:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.update_layout(template="plotly_dark",
                              title=f"No data for {ticker}. Run feature pipeline first.")
            return fig

        df = add_regime_labels(df, ticker)

        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        return build_price_chart(df, ticker)

    @app.callback(
        Output("signal-chart", "figure"),
        Output("pnl-chart", "figure"),
        Input("signal-ticker", "value"),
        Input("signal-model", "value"),
    )
    def update_signals(ticker, model):
        import plotly.graph_objects as go
        df = load_feature_df(ticker, cfg)
        if df is None:
            empty = go.Figure()
            empty.update_layout(template="plotly_dark", title="No data available.")
            return empty, empty

        cv_df = load_cv_results(cfg)
        if cv_df is None:
            empty = go.Figure()
            empty.update_layout(template="plotly_dark",
                                title="No CV results. Run walk_forward.py first.")
            return empty, empty

        model_cv = cv_df[cv_df["model"] == model]
        if model_cv.empty:
            empty = go.Figure()
            empty.update_layout(template="plotly_dark", title=f"No results for model '{model}'.")
            return empty, empty

        # Generate signals from HMM volatility regime:
        # Low-vol regime (0) → buy, High-vol regime (1) → sell/cash
        df = add_regime_labels(df, ticker)
        predicted_returns = _regime_signal(df)

        signals_df = generate_signals(
            predicted_returns,
            buy_threshold=0.5,
            sell_threshold=-0.5,
        )
        backtest_df = backtest_signals(signals_df, df["log_return"])

        signal_fig = build_signal_chart(df, signals_df, ticker)
        pnl_fig = build_pnl_chart(backtest_df)
        return signal_fig, pnl_fig
