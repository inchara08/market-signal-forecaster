"""Regime-aware ARIMA/GARCH model for return forecasting."""
import numpy as np
import pandas as pd
import warnings
import joblib
import os


class RegimeAwareARIMA:
    """
    Fits a separate ARIMA (optionally wrapped with GARCH) per volatility regime.
    Uses pmdarima.auto_arima for automatic (p,d,q) selection via AIC.
    Applies Engle's ARCH-LM test — if significant, wraps with GARCH(1,1).
    """

    def __init__(self, max_p: int = 5, max_q: int = 5, max_d: int = 2,
                 arch_significance: float = 0.05):
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.arch_significance = arch_significance
        self.models: dict = {}      # regime → fitted model
        self.use_garch: dict = {}   # regime → bool
        self.orders: dict = {}      # regime → (p, d, q)

    def fit(self, df: pd.DataFrame, regime_col: str = "volatility_regime") -> "RegimeAwareARIMA":
        from pmdarima import auto_arima

        returns = df["log_return"].dropna()
        regimes = df.loc[returns.index, regime_col]

        for regime in sorted(regimes.unique()):
            regime = int(regime)
            mask = regimes == regime
            regime_returns = returns[mask]

            if len(regime_returns) < 50:
                print(f"[arima] Regime {regime}: too few samples ({len(regime_returns)}), skipping")
                continue

            print(f"[arima] Regime {regime}: fitting on {len(regime_returns)} samples...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = auto_arima(
                    regime_returns,
                    max_p=self.max_p, max_q=self.max_q, max_d=self.max_d,
                    information_criterion="aic",
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                )

            self.orders[regime] = model.order
            residuals = model.resid()

            # Engle ARCH-LM test on residuals
            if self._arch_lm_significant(residuals):
                print(f"[arima] Regime {regime}: ARCH effects detected → wrapping with GARCH(1,1)")
                self.use_garch[regime] = True
                self.models[regime] = (model, self._fit_garch(residuals))
            else:
                self.use_garch[regime] = False
                self.models[regime] = model

            print(f"[arima] Regime {regime}: order={model.order}, GARCH={self.use_garch[regime]}")

        return self

    def predict(self, regime: int, steps: int = 1) -> np.ndarray:
        """Return point forecast for the given regime."""
        if regime not in self.models:
            return np.zeros(steps)

        if self.use_garch.get(regime, False):
            arima_model, _ = self.models[regime]
        else:
            arima_model = self.models[regime]

        return arima_model.predict(n_periods=steps)

    def predict_from_df(self, df: pd.DataFrame, regime_col: str = "volatility_regime") -> pd.Series:
        """Produce one-step-ahead forecasts for each row based on its regime."""
        preds = pd.Series(np.nan, index=df.index)
        returns = df["log_return"].dropna()
        regimes = df.loc[returns.index, regime_col]

        for regime in sorted(regimes.unique()):
            mask = regimes == regime
            idx = mask.index[mask]
            forecast = self.predict(int(regime), steps=1)
            preds.loc[idx] = float(np.asarray(forecast).flat[0])

        return preds

    def get_regime_params(self) -> dict:
        return {r: {"order": self.orders.get(r), "garch": self.use_garch.get(r)}
                for r in self.models}

    def _arch_lm_significant(self, residuals: np.ndarray) -> bool:
        try:
            from arch.unitroot import engle_granger
            from statsmodels.stats.diagnostic import het_arch
            stat, p_value, _, _ = het_arch(residuals, nlags=5)
            return p_value < self.arch_significance
        except Exception:
            return False

    def _fit_garch(self, residuals: np.ndarray):
        from arch import arch_model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(residuals * 100, vol="Garch", p=1, q=1, rescale=False)
            return am.fit(disp="off", show_warning=False)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "RegimeAwareARIMA":
        return joblib.load(path)
