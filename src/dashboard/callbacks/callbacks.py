"""Dash callback registrations."""
import os
import pandas as pd
from dash import Input, Output, callback
import yaml

from src.dashboard.layouts.overview import build_price_chart
from src.dashboard.layouts.signal_view import build_signal_chart, build_pnl_chart
from src.signals.signal_generator import generate_signals, backtest_signals


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


def register_callbacks(app, cfg: dict, cache) -> None:

    @app.callback(
        Output("overview-chart", "figure"),
        Input("overview-ticker", "value"),
        Input("overview-daterange", "start_date"),
        Input("overview-daterange", "end_date"),
    )
    @cache.memoize(timeout=cfg["dashboard"]["cache_timeout"])
    def update_overview(ticker, start_date, end_date):
        df = load_feature_df(ticker, cfg)
        if df is None:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.update_layout(template="plotly_dark",
                              title=f"No data for {ticker}. Run feature pipeline first.")
            return fig
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
    @cache.memoize(timeout=cfg["dashboard"]["cache_timeout"])
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

        # Use mean predicted return from CV as a proxy for signal generation
        # In a full system this would use live model inference
        if "ic" in model_cv.columns:
            predicted_returns = df["log_return"].shift(1).fillna(0) * model_cv["ic"].mean()
        else:
            predicted_returns = df["log_return"].shift(1).fillna(0)

        signals_df = generate_signals(
            predicted_returns,
            buy_threshold=cfg["signals"]["buy_threshold"],
            sell_threshold=cfg["signals"]["sell_threshold"],
        )
        backtest_df = backtest_signals(signals_df, df["log_return"])

        signal_fig = build_signal_chart(df, signals_df, ticker)
        pnl_fig = build_pnl_chart(backtest_df)
        return signal_fig, pnl_fig
