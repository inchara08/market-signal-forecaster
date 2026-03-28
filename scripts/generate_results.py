"""
Generate static result charts and copy CV results for the GitHub portfolio.

Outputs:
  results/images/regime_overlay.png   — AAPL price + HMM volatility regime shading
  results/images/cv_metrics.png       — Walk-forward directional accuracy & MAPE improvement
  results/images/cumulative_pnl.png   — Ensemble strategy vs buy-and-hold
  results/cv_results.csv              — Copy of walk-forward results
"""
import os
import sys
import shutil
import warnings

warnings.filterwarnings("ignore")

# Run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS_DIR = "results"
IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

DARK_BG = "#0d1117"
DARK_PAPER = "#161b22"
GRID_COLOR = "#30363d"
TEXT_COLOR = "#c9d1d9"
COLORS = {
    "price": "#58a6ff",
    "arima": "#f78166",
    "lstm": "#7ee787",
    "ensemble": "#e3b341",
    "buyhold": "#8b949e",
    "regime_low": "rgba(46,160,67,0.18)",
    "regime_high": "rgba(248,81,73,0.18)",
}

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_PAPER,
    font=dict(color=TEXT_COLOR, family="monospace", size=13),
    xaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
    yaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
    margin=dict(l=60, r=40, t=60, b=50),
)


# ── Chart 1: Regime Overlay ────────────────────────────────────────────────────

def chart_regime_overlay():
    print("[1/3] Generating regime overlay chart...")
    from src.models.regime import RegimeDetector

    df = pd.read_parquet("data/processed/AAPL_features.parquet")

    rd = RegimeDetector(n_components=2)
    df = rd.fit_predict(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"],
        name="AAPL Close",
        line=dict(color=COLORS["price"], width=1.5),
    ))

    # Regime shading
    regime = df["volatility_regime"]
    prev = int(regime.iloc[0])
    start = df.index[0]
    color_map = {0: COLORS["regime_low"], 1: COLORS["regime_high"]}
    for i in range(1, len(regime)):
        curr = int(regime.iloc[i])
        if curr != prev or i == len(regime) - 1:
            fig.add_vrect(
                x0=start, x1=df.index[i],
                fillcolor=color_map[prev],
                layer="below", line_width=0,
            )
            start = df.index[i]
            prev = curr

    # Regime legend proxies
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=12, color=COLORS["regime_low"].replace("0.18", "0.7"), symbol="square"),
                             name="Low Volatility Regime"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=12, color=COLORS["regime_high"].replace("0.18", "0.7"), symbol="square"),
                             name="High Volatility Regime"))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="AAPL Price — HMM Volatility Regime Detection (2018–2024)", x=0.5, font=dict(size=15)),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        width=1200, height=500,
    )

    path = os.path.join(IMAGES_DIR, "regime_overlay.png")
    fig.write_image(path, scale=2)
    print(f"   → {path}")


# ── Chart 2: Walk-Forward CV Metrics ──────────────────────────────────────────

def chart_cv_metrics():
    print("[2/3] Generating walk-forward metrics chart...")
    cv = pd.read_csv("data/processed/cv_results.csv")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Directional Accuracy by Fold (%)", "MAPE Improvement vs. Naive Baseline (%)"],
        horizontal_spacing=0.1,
    )

    model_colors = {"arima": COLORS["arima"], "lstm": COLORS["lstm"], "ensemble": COLORS["ensemble"]}

    for model, color in model_colors.items():
        sub = cv[cv["model"] == model].sort_values("fold")
        fig.add_trace(go.Scatter(
            x=sub["fold"], y=sub["directional_accuracy"],
            name=model.upper(), line=dict(color=color, width=2),
            mode="lines+markers", marker=dict(size=6),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sub["fold"], y=sub["mape_improvement_vs_baseline"],
            name=model.upper(), line=dict(color=color, width=2),
            mode="lines+markers", marker=dict(size=6),
            showlegend=False,
        ), row=1, col=2)

    # 50% random baseline
    fig.add_hline(y=50, line_dash="dot", line_color="#8b949e",
                  annotation_text="Random (50%)", annotation_position="bottom right",
                  row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#8b949e",
                  annotation_text="Baseline (0%)", annotation_position="bottom right",
                  row=1, col=2)

    fig.update_layout(
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k not in ("xaxis", "yaxis")},
        title=dict(text="Walk-Forward Cross-Validation Results — AAPL (15 Folds)", x=0.5, font=dict(size=15)),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        width=1200, height=450,
        xaxis=dict(title="Fold", gridcolor=GRID_COLOR),
        xaxis2=dict(title="Fold", gridcolor=GRID_COLOR),
        yaxis=dict(title="Directional Accuracy (%)", gridcolor=GRID_COLOR),
        yaxis2=dict(title="MAPE Improvement (%)", gridcolor=GRID_COLOR),
    )

    path = os.path.join(IMAGES_DIR, "cv_metrics.png")
    fig.write_image(path, scale=2)
    print(f"   → {path}")


# ── Chart 3: Cumulative P&L ────────────────────────────────────────────────────

def chart_cumulative_pnl():
    print("[3/3] Generating cumulative P&L chart...")
    cv = pd.read_csv("data/processed/cv_results.csv")
    df = pd.read_parquet("data/processed/AAPL_features.parquet")

    # Build a per-date signal from ensemble IC across val periods
    ensemble_cv = cv[cv["model"] == "ensemble"].sort_values("fold")

    # Collect val-period actual returns + ensemble directional signal
    val_rows = []
    for _, row in ensemble_cv.iterrows():
        period = df.loc[row["val_start"]:row["val_end"], "log_return"].dropna()
        ic = row["ic"] if not np.isnan(row["ic"]) else 0
        # Signal: +1 if IC > 0 (model has positive skill), else -1
        signal = 1 if ic >= 0 else -1
        for date, ret in period.items():
            val_rows.append({"date": date, "return": ret, "signal": signal})

    if not val_rows:
        print("   ⚠ No val rows found, skipping P&L chart")
        return

    pnl_df = pd.DataFrame(val_rows).set_index("date").sort_index()
    pnl_df["strategy_return"] = pnl_df["signal"].shift(1).fillna(1) * pnl_df["return"]
    pnl_df["cumulative_strategy"] = (1 + pnl_df["strategy_return"]).cumprod()
    pnl_df["cumulative_buyhold"] = (1 + pnl_df["return"]).cumprod()

    final_strat = pnl_df["cumulative_strategy"].iloc[-1]
    final_bh = pnl_df["cumulative_buyhold"].iloc[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pnl_df.index, y=pnl_df["cumulative_strategy"],
        name=f"Ensemble Strategy  ({final_strat:.2f}×)",
        line=dict(color=COLORS["ensemble"], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=pnl_df.index, y=pnl_df["cumulative_buyhold"],
        name=f"Buy & Hold  ({final_bh:.2f}×)",
        line=dict(color=COLORS["buyhold"], width=1.8, dash="dash"),
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#30363d")

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Cumulative Returns: Ensemble Signal Strategy vs Buy & Hold (Val Periods)", x=0.5, font=dict(size=15)),
        xaxis_title="Date",
        yaxis_title="Cumulative Return (×)",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        width=1200, height=400,
    )

    path = os.path.join(IMAGES_DIR, "cumulative_pnl.png")
    fig.write_image(path, scale=2)
    print(f"   → {path}")


# ── Copy CV Results ────────────────────────────────────────────────────────────

def copy_cv_results():
    src = "data/processed/cv_results.csv"
    dst = os.path.join(RESULTS_DIR, "cv_results.csv")
    shutil.copy(src, dst)
    print(f"[+] Copied {src} → {dst}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Generating portfolio result assets ===\n")
    chart_regime_overlay()
    chart_cv_metrics()
    chart_cumulative_pnl()
    copy_cv_results()
    print("\nDone. Files written to results/")
