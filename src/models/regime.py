"""Volatility regime detection via Gaussian HMM."""
import numpy as np
import pandas as pd
import joblib
import os


class RegimeDetector:
    """
    Fit a Gaussian HMM on [log_return, realized_vol_10d] to label
    market regimes (0 = low-vol, 1 = high-vol).
    """

    def __init__(self, n_components: int = 2, n_iter: int = 200, covariance_type: str = "full"):
        self.n_components = n_components
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.model = None
        self._state_map = None  # maps HMM state → regime label (low=0, high=1)

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        from hmmlearn.hmm import GaussianHMM

        X = self._prepare(df)
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=42,
        )
        self.model.fit(X)

        # Map states: state with lower mean volatility → regime 0
        vol_idx = 1  # index of realized_vol_10d in feature matrix
        mean_vols = self.model.means_[:, vol_idx]
        sorted_states = np.argsort(mean_vols)
        self._state_map = {state: label for label, state in enumerate(sorted_states)}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return regime labels array aligned with df index."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._prepare(df)
        raw_states = self.model.predict(X)
        return np.array([self._state_map[s] for s in raw_states])

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and add 'volatility_regime' column to df."""
        self.fit(df)
        df = df.copy()
        df["volatility_regime"] = self.predict(df)
        return df

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        cols = ["log_return", "realized_vol_10d"]
        X = df[cols].copy()
        X = X.ffill().fillna(0)
        return X.values

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "state_map": self._state_map}, path)

    @classmethod
    def load(cls, path: str) -> "RegimeDetector":
        data = joblib.load(path)
        obj = cls()
        obj.model = data["model"]
        obj._state_map = data["state_map"]
        return obj
