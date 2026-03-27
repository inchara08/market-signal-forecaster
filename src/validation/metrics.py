"""Evaluation metrics: regression + financial signal metrics."""
import numpy as np
import pandas as pd
from scipy import stats


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    return np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = ~np.isnan(actual) & ~np.isnan(predicted) & (actual != 0)
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    denom = (np.abs(actual[mask]) + np.abs(predicted[mask])) / 2
    nonzero = denom != 0
    return np.mean(np.abs(actual[mask][nonzero] - predicted[mask][nonzero]) / denom[nonzero]) * 100


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    return np.mean(np.abs(actual[mask] - predicted[mask]))


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of predictions with correct sign (trend direction)."""
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    return np.mean(np.sign(actual[mask]) == np.sign(predicted[mask])) * 100


def information_coefficient(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Spearman rank correlation between predicted and actual returns."""
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if mask.sum() < 3:
        return np.nan
    corr, _ = stats.spearmanr(actual[mask], predicted[mask])
    return corr


def icir(ic_series: pd.Series) -> float:
    """Information Coefficient Information Ratio: mean(IC) / std(IC)."""
    if ic_series.std() == 0:
        return np.nan
    return ic_series.mean() / ic_series.std()


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Sharpe ratio of a signal-driven return series."""
    if returns.std() == 0:
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def mape_improvement_vs_baseline(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Percentage MAPE improvement over a naive persistence baseline."""
    baseline_pred = np.roll(actual, 1)
    baseline_pred[0] = actual[0]
    baseline_mape = mape(actual, baseline_pred)
    model_mape = mape(actual, predicted)
    if baseline_mape == 0:
        return 0.0
    return (baseline_mape - model_mape) / baseline_mape * 100


def compute_all(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    return {
        "rmse": rmse(actual, predicted),
        "mae": mae(actual, predicted),
        "mape": mape(actual, predicted),
        "smape": smape(actual, predicted),
        "directional_accuracy": directional_accuracy(actual, predicted),
        "ic": information_coefficient(actual, predicted),
        "mape_improvement_vs_baseline": mape_improvement_vs_baseline(actual, predicted),
    }
