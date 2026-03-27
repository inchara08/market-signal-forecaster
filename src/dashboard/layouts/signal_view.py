"""Dashboard page 3: Signal View — trade signals + cumulative P&L."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import dcc, html


SIGNAL_COLORS = {1: "#27ae60", -1: "#e74c3c", 0: "#f39c12"}
SIGNAL_SYMBOLS = {1: "triangle-up", -1: "triangle-down", 0: "circle"}


def build_signal_chart(df: pd.DataFrame, signals_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Price chart with signal markers overlaid."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)

    # Price line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"], name="Close",
        line=dict(color="#89b4fa", width=1.5),
    ), row=1, col=1)

    # Signal markers
    for sig_val, sig_name in [(1, "Buy"), (-1, "Sell")]:
        mask = signals_df["signal"] == sig_val
        if mask.any():
            idx = signals_df.index[mask]
            prices = df.loc[df.index.isin(idx), "close"]
            fig.add_trace(go.Scatter(
                x=prices.index, y=prices.values,
                mode="markers",
                name=sig_name,
                marker=dict(
                    symbol=SIGNAL_SYMBOLS[sig_val],
                    color=SIGNAL_COLORS[sig_val],
                    size=10,
                ),
            ), row=1, col=1)

    # Confidence
    if "confidence" in signals_df.columns:
        fig.add_trace(go.Bar(
            x=signals_df.index, y=signals_df["confidence"],
            name="Signal Confidence",
            marker_color="#cba6f7",
        ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=f"{ticker} — Trading Signals", x=0.5),
        height=550, margin=dict(l=50, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.02),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Confidence", row=2, col=1)
    return fig


def build_pnl_chart(backtest_df: pd.DataFrame) -> go.Figure:
    """Cumulative P&L: strategy vs buy-and-hold."""
    fig = go.Figure()
    if "cumulative_strategy" in backtest_df.columns:
        fig.add_trace(go.Scatter(
            x=backtest_df.index, y=backtest_df["cumulative_strategy"],
            name="Signal Strategy", line=dict(color="#a6e3a1", width=2),
        ))
    if "cumulative_buyhold" in backtest_df.columns:
        fig.add_trace(go.Scatter(
            x=backtest_df.index, y=backtest_df["cumulative_buyhold"],
            name="Buy & Hold", line=dict(color="#89dceb", width=1.5, dash="dash"),
        ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray")
    fig.update_layout(
        template="plotly_dark",
        title="Cumulative Returns: Strategy vs Buy & Hold",
        xaxis_title="Date", yaxis_title="Cumulative Return (×)",
        height=350, margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def layout(tickers: list[str]) -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Label("Ticker"),
                dcc.Dropdown(
                    id="signal-ticker",
                    options=[{"label": t, "value": t} for t in tickers],
                    value=tickers[0],
                    clearable=False,
                    style={"color": "#000"},
                ),
            ], width=3),
            dbc.Col([
                dbc.Label("Model"),
                dcc.Dropdown(
                    id="signal-model",
                    options=[
                        {"label": "Ensemble", "value": "ensemble"},
                        {"label": "ARIMA", "value": "arima"},
                        {"label": "LSTM", "value": "lstm"},
                    ],
                    value="ensemble",
                    clearable=False,
                    style={"color": "#000"},
                ),
            ], width=3),
        ], className="mb-3"),
        dcc.Graph(id="signal-chart"),
        dcc.Graph(id="pnl-chart"),
    ], className="p-3")
