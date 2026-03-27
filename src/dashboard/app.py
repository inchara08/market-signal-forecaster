"""Dash multi-page application entrypoint."""
import os
import yaml
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
from flask_caching import Cache

from src.dashboard.layouts import overview, model_perf, signal_view
from src.dashboard.callbacks.callbacks import register_callbacks, load_cv_results


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def create_app(config_path: str = "config/config.yaml") -> dash.Dash:
    cfg = load_config(config_path)
    tickers = cfg["tickers"]

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        suppress_callback_exceptions=True,
        title="Market Signal Forecaster",
    )

    # Flask-Caching
    cache = Cache(app.server, config={
        "CACHE_TYPE": "SimpleCache",
        "CACHE_DEFAULT_TIMEOUT": cfg["dashboard"]["cache_timeout"],
    })

    # Load CV results once at startup
    cv_df = load_cv_results(cfg)

    # Navigation
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Market Overview", href="/", active="exact")),
            dbc.NavItem(dbc.NavLink("Model Performance", href="/performance", active="exact")),
            dbc.NavItem(dbc.NavLink("Signal View", href="/signals", active="exact")),
        ],
        brand="Market Signal Forecaster",
        brand_href="/",
        color="dark",
        dark=True,
        className="mb-2",
    )

    app.layout = html.Div([
        dcc.Location(id="url"),
        navbar,
        html.Div(id="page-content"),
    ])

    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def display_page(pathname):
        if pathname == "/performance":
            return model_perf.layout(cv_df)
        elif pathname == "/signals":
            return signal_view.layout(tickers)
        else:
            return overview.layout(tickers)

    register_callbacks(app, cfg, cache)
    return app


if __name__ == "__main__":
    cfg = load_config()
    app = create_app()
    app.run(
        host=cfg["dashboard"]["host"],
        port=cfg["dashboard"]["port"],
        debug=cfg["dashboard"]["debug"],
    )
