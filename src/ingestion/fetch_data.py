"""Download OHLCV data via yfinance and cache to parquet."""
import os
import yaml
import yfinance as yf
import pandas as pd


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_ticker(
    ticker: str,
    start: str,
    end: str,
    raw_dir: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download OHLCV for a single ticker; return cached parquet if available."""
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{ticker}.parquet")

    if os.path.exists(path) and not force_refresh:
        return pd.read_parquet(path)

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    df["ticker"] = ticker
    df.to_parquet(path)
    print(f"[fetch] {ticker}: {len(df)} rows saved to {path}")
    return df


def fetch_all(config_path: str = "config/config.yaml", force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    cfg = load_config(config_path)
    results = {}
    for ticker in cfg["tickers"]:
        results[ticker] = fetch_ticker(
            ticker,
            start=cfg["data"]["start_date"],
            end=cfg["data"]["end_date"],
            raw_dir=cfg["data"]["raw_dir"],
            force_refresh=force_refresh,
        )
    return results


if __name__ == "__main__":
    data = fetch_all()
    for ticker, df in data.items():
        print(f"{ticker}: {df.shape}, {df.index.min().date()} → {df.index.max().date()}")
