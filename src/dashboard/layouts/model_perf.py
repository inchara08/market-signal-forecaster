"""Dashboard page 2: Model Performance — walk-forward metrics & IC chart."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table


def build_metrics_table(cv_df: pd.DataFrame) -> dash_table.DataTable:
    cols_show = ["fold", "model", "val_start", "val_end", "rmse", "mape",
                 "directional_accuracy", "mape_improvement_vs_baseline", "ic"]
    display_df = cv_df[[c for c in cols_show if c in cv_df.columns]].copy()

    for col in ["rmse", "mape", "directional_accuracy", "mape_improvement_vs_baseline", "ic"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)

    return dash_table.DataTable(
        data=display_df.to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in display_df.columns],
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": "#1e1e2e", "color": "#cdd6f4", "fontSize": 13,
                    "padding": "8px", "fontFamily": "monospace"},
        style_header={"backgroundColor": "#313244", "fontWeight": "bold", "color": "#89b4fa"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#181825"},
            {"if": {"filter_query": "{model} = 'ensemble'"},
             "backgroundColor": "#1e3a5f", "fontWeight": "bold"},
        ],
        page_size=20,
        sort_action="native",
        filter_action="native",
    )


def build_ic_chart(cv_df: pd.DataFrame) -> go.Figure:
    """Rolling IC over folds for each model."""
    fig = go.Figure()
    for model in cv_df["model"].unique():
        sub = cv_df[cv_df["model"] == model].sort_values("fold")
        fig.add_trace(go.Scatter(
            x=sub["fold"], y=sub["ic"],
            name=model.upper(), mode="lines+markers",
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        template="plotly_dark", title="Information Coefficient by Fold",
        xaxis_title="Fold", yaxis_title="IC (Spearman)",
        height=350, margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def build_mape_chart(cv_df: pd.DataFrame) -> go.Figure:
    """MAPE by fold for each model."""
    fig = go.Figure()
    for model in cv_df["model"].unique():
        sub = cv_df[cv_df["model"] == model].sort_values("fold")
        fig.add_trace(go.Scatter(
            x=sub["fold"], y=sub["mape"],
            name=model.upper(), mode="lines+markers",
        ))
    fig.update_layout(
        template="plotly_dark", title="MAPE by Fold",
        xaxis_title="Fold", yaxis_title="MAPE (%)",
        height=350, margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def layout(cv_df: pd.DataFrame | None) -> html.Div:
    if cv_df is None or cv_df.empty:
        return html.Div(
            dbc.Alert("No cross-validation results found. Run walk_forward.py first.", color="warning"),
            className="p-3",
        )

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=build_ic_chart(cv_df)), width=6),
            dbc.Col(dcc.Graph(figure=build_mape_chart(cv_df)), width=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.H5("Walk-Forward Validation Results", className="text-info mb-2"),
                build_metrics_table(cv_df),
            ]),
        ]),
    ], className="p-3")
