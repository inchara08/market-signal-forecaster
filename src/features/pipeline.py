"""Assemble all features in one pass and save to processed parquet."""
import os
import yaml
import pandas as pd

from src.preprocessing.cleaner import clean
from src.features.returns import add_return_features
from src.features.volatility import add_volatility_features
from src.features.technical import add_technical_features
from src.features.statistical import add_statistical_features
from src.features.constants import FEATURE_COLS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline on a cleaned OHLCV DataFrame."""
    df = clean(df)
    df = add_return_features(df)
    df = add_volatility_features(df)
    df = add_technical_features(df)
    df = add_statistical_features(df)
    # Drop warmup rows (first 100 rows where rolling features are unreliable)
    df = df.iloc[100:].copy()
    return df


def run_pipeline(config_path: str = "config/config.yaml") -> dict[str, pd.DataFrame]:
    from src.ingestion.fetch_data import fetch_all, load_config
    cfg = load_config(config_path)
    processed_dir = cfg["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    raw_data = fetch_all(config_path)
    results = {}

    for ticker, df in raw_data.items():
        print(f"[pipeline] Processing {ticker}...")
        feat_df = build_features(df)
        out_path = os.path.join(processed_dir, f"{ticker}_features.parquet")
        feat_df.to_parquet(out_path)
        n_features = len([c for c in feat_df.columns if c in FEATURE_COLS])
        print(f"[pipeline] {ticker}: {len(feat_df)} rows, {n_features} features → {out_path}")
        results[ticker] = feat_df

    return results


if __name__ == "__main__":
    from src.features.constants import FEATURE_COLS as _FC
    dfs = run_pipeline()
    for ticker, df in dfs.items():
        feat_cols = [c for c in df.columns if c in _FC]
        null_pct = df[feat_cols].isnull().mean().mean() * 100
        print(f"{ticker}: {len(feat_cols)} features, {null_pct:.1f}% null rate")
