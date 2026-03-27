"""Walk-forward cross-validation engine for ARIMA and LSTM models."""
import os
import numpy as np
import pandas as pd
import yaml

from src.preprocessing.splitter import WalkForwardSplitter
from src.models.regime import RegimeDetector
from src.models.arima_model import RegimeAwareARIMA
from src.models.lstm_model import LSTMForecaster
from src.validation.metrics import compute_all
from src.features.constants import FEATURE_COLS


def run_walk_forward(
    df: pd.DataFrame,
    config_path: str = "config/config.yaml",
    output_path: str = "data/processed/cv_results.csv",
) -> pd.DataFrame:
    """
    Run expanding-window walk-forward validation.
    Returns a DataFrame with per-fold metrics for ARIMA, LSTM, and an ensemble.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    wf_cfg = cfg["walk_forward"]
    splitter = WalkForwardSplitter(
        initial_train_size=wf_cfg["initial_train_size"],
        step_size=wf_cfg["step_size"],
        val_size=wf_cfg["val_size"],
        expanding_window=wf_cfg["expanding_window"],
    )

    feature_cols = [c for c in FEATURE_COLS if c in df.columns and c != "log_return"]
    target_col = "log_return"

    df_clean = df.dropna(subset=[target_col] + feature_cols).copy()
    n = len(df_clean)

    all_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(n)):
        print(f"[walk_forward] Fold {fold_idx + 1}: train={len(train_idx)}, val={len(val_idx)}")

        train_df = df_clean.iloc[train_idx]
        val_df = df_clean.iloc[val_idx]

        actual = val_df[target_col].values

        # ── ARIMA ──────────────────────────────────────────────────────────────
        try:
            regime_det = RegimeDetector(
                n_components=cfg["regime"]["n_components"],
                n_iter=cfg["regime"]["n_iter"],
            )
            train_df = regime_det.fit_predict(train_df)
            val_df = val_df.copy()
            val_df["volatility_regime"] = regime_det.predict(val_df)

            arima = RegimeAwareARIMA(
                max_p=cfg["arima"]["max_p"],
                max_q=cfg["arima"]["max_q"],
                arch_significance=cfg["arima"]["arch_lm_significance"],
            )
            arima.fit(train_df)
            arima_preds = arima.predict_from_df(val_df).values
        except Exception as e:
            print(f"  [arima] fold {fold_idx + 1} failed: {e}")
            arima_preds = np.full(len(val_idx), np.nan)

        # ── LSTM ───────────────────────────────────────────────────────────────
        try:
            lstm_cfg = cfg["lstm"]
            lstm = LSTMForecaster(
                lookback=lstm_cfg["lookback_window"],
                units_1=lstm_cfg["units_layer1"],
                units_2=lstm_cfg["units_layer2"],
                dense_units=lstm_cfg["dense_units"],
                dropout=lstm_cfg["dropout"],
                batch_size=lstm_cfg["batch_size"],
                max_epochs=lstm_cfg["max_epochs"],
                patience=lstm_cfg["patience"],
            )
            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values
            X_val = val_df[feature_cols].values
            y_val = val_df[target_col].values

            lstm.fit(X_train, y_train, X_val, y_val)

            # Predict on val using full context (train + val) for sequence creation
            X_full = df_clean.iloc[: val_idx[-1] + 1][feature_cols].values
            all_preds = lstm.predict(X_full)
            lstm_preds = all_preds[val_idx]
        except Exception as e:
            print(f"  [lstm] fold {fold_idx + 1} failed: {e}")
            lstm_preds = np.full(len(val_idx), np.nan)

        # ── Ensemble (equal weight, skip NaN) ─────────────────────────────────
        ensemble_preds = np.where(
            np.isnan(arima_preds) | np.isnan(lstm_preds),
            np.where(np.isnan(arima_preds), lstm_preds, arima_preds),
            0.5 * arima_preds + 0.5 * lstm_preds,
        )

        # ── Metrics ───────────────────────────────────────────────────────────
        for model_name, preds in [("arima", arima_preds), ("lstm", lstm_preds), ("ensemble", ensemble_preds)]:
            m = compute_all(actual, preds)
            m["fold"] = fold_idx + 1
            m["model"] = model_name
            m["train_size"] = len(train_idx)
            m["val_size"] = len(val_idx)
            m["val_start"] = df_clean.index[val_idx[0]]
            m["val_end"] = df_clean.index[val_idx[-1]]
            all_results.append(m)

        print(f"  [ensemble] dir_acc={ensemble_preds[~np.isnan(ensemble_preds)].shape}")

    results_df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n[walk_forward] Results saved to {output_path}")

    # Summary
    summary = results_df.groupby("model")[["rmse", "mape", "directional_accuracy", "mape_improvement_vs_baseline"]].mean()
    print("\n=== Walk-Forward Summary (mean across folds) ===")
    print(summary.to_string())
    return results_df


if __name__ == "__main__":
    import sys
    from src.features.pipeline import run_pipeline

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    dfs = run_pipeline()
    results = run_walk_forward(dfs[ticker])
