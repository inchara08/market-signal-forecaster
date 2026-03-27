"""
Market Signal Forecaster — orchestration entrypoint.

Usage:
    python main.py pipeline          # fetch data + build features
    python main.py validate AAPL     # run walk-forward CV on a ticker
    python main.py dashboard         # launch the Dash dashboard
    python main.py all AAPL          # pipeline + validate + dashboard
"""
import sys
import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline():
    from src.features.pipeline import run_pipeline as _run
    print("=== Running feature pipeline ===")
    dfs = _run()
    print(f"Done. Processed {len(dfs)} tickers.")
    return dfs


def run_validation(ticker: str):
    from src.features.pipeline import run_pipeline as _run
    from src.validation.walk_forward import run_walk_forward
    print(f"=== Running walk-forward validation for {ticker} ===")
    dfs = _run()
    if ticker not in dfs:
        print(f"Ticker {ticker} not found. Available: {list(dfs.keys())}")
        return
    results = run_walk_forward(dfs[ticker])
    ensemble = results[results["model"] == "ensemble"]
    mean_improvement = ensemble["mape_improvement_vs_baseline"].mean()
    mean_dir_acc = ensemble["directional_accuracy"].mean()
    print(f"\n=== Final Results ===")
    print(f"Ensemble mean MAPE improvement vs. baseline: {mean_improvement:.1f}%")
    print(f"Ensemble mean directional accuracy:          {mean_dir_acc:.1f}%")


def run_dashboard():
    from src.dashboard.app import create_app
    cfg = load_config()
    print(f"=== Launching dashboard at http://localhost:{cfg['dashboard']['port']} ===")
    app = create_app()
    app.run(
        host=cfg["dashboard"]["host"],
        port=cfg["dashboard"]["port"],
        debug=True,
    )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "dashboard"
    ticker = sys.argv[2] if len(sys.argv) > 2 else "AAPL"

    if command == "pipeline":
        run_pipeline()
    elif command == "validate":
        run_validation(ticker)
    elif command == "dashboard":
        run_dashboard()
    elif command == "all":
        run_pipeline()
        run_validation(ticker)
        run_dashboard()
    else:
        print(__doc__)
