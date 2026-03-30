"""Dashboard page 1: Market Overview — candlestick, regime heatmap, volume."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import dcc, html


REGIME_COLORS = {0: "rgba(39,174,96,0.12)", 1: "rgba(231,76,60,0.12)"}
REGIME_LABELS = {0: "Low Volatility", 1: "High Volatility"}


def build_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Candlestick + SMA + Bollinger bands + regime shading + volume."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name=ticker, showlegend=False,
    ), row=1, col=1)

    # SMA 20
    if "price_to_sma20" in df.columns:
        sma20 = df["close"] / df["price_to_sma20"]
        fig.add_trace(go.Scatter(
            x=df.index, y=sma20, name="SMA 20",
            line=dict(color="royalblue", width=1.2),
        ), row=1, col=1)

    # Bollinger Bands
    if "bb_pct_b" in df.columns:
        sma = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        fig.add_trace(go.Scatter(
            x=df.index, y=sma + 2 * std,
            name="BB Upper", line=dict(color="gray", width=0.8, dash="dash"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=sma - 2 * std,
            name="BB Lower", line=dict(color="gray", width=0.8, dash="dash"),
            fill="tonexty", fillcolor="rgba(128,128,128,0.08)",
        ), row=1, col=1)

    # Regime shading
    if "volatility_regime" in df.columns:
        _add_regime_shading(fig, df, row=1)

    # Volume bars
    ret = df["close"].pct_change().fillna(0).values
    colors = ["#26a69a" if r >= 0 else "#ef5350" for r in ret]
    fig.add_trace(go.Bar(
        x=list(df.index), y=df["volume"].tolist(),
        name="Volume",
        marker=dict(color=colors, opacity=0.8),
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=50, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.02),
        title=dict(text=f"{ticker} — Price & Volume", x=0.5),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def _add_regime_shading(fig: go.Figure, df: pd.DataFrame, row: int = 1) -> None:
    """Add colored background shapes for each regime period (price subplot only)."""
    if "volatility_regime" not in df.columns:
        return
    regime = df["volatility_regime"]
    prev_val = regime.iloc[0]
    start = df.index[0]

    # yref="y domain" limits the shape to the price subplot's y-axis extent only
    yref = "y domain" if row == 1 else f"y{row} domain"

    for i in range(1, len(regime)):
        if regime.iloc[i] != prev_val or i == len(regime) - 1:
            fig.add_shape(
                type="rect",
                x0=start, x1=df.index[i],
                y0=0, y1=1,
                xref="x", yref=yref,
                fillcolor=REGIME_COLORS.get(prev_val, "rgba(0,0,0,0)"),
                layer="below", line_width=0,
            )
            start = df.index[i]
            prev_val = regime.iloc[i]


def layout(tickers: list[str]) -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Label("Ticker"),
                dcc.Dropdown(
                    id="overview-ticker",
                    options=[{"label": t, "value": t} for t in tickers],
                    value=tickers[0],
                    clearable=False,
                    style={"color": "#000"},
                ),
            ], width=3),
            dbc.Col([
                dbc.Label("Date Range"),
                dcc.DatePickerRange(
                    id="overview-daterange",
                    display_format="YYYY-MM-DD",
                ),
            ], width=5),
        ], className="mb-3"),
        dcc.Graph(id="overview-chart", style={"height": "600px"}),
    ], className="p-3")
