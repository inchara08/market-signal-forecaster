"""LSTM forecaster for multi-step return prediction."""
import numpy as np
import pandas as pd
import os
import joblib


class LSTMForecaster:
    """
    Two-layer LSTM with regime embedding.
    Architecture: LSTM(128) → LSTM(64) → Dense(32) → Dense(1)
    Loss: Huber, Optimizer: Adam with gradient clipping
    """

    def __init__(
        self,
        lookback: int = 60,
        units_1: int = 128,
        units_2: int = 64,
        dense_units: int = 32,
        dropout: float = 0.2,
        recurrent_dropout: float = 0.1,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 15,
        clip_norm: float = 1.0,
    ):
        self.lookback = lookback
        self.units_1 = units_1
        self.units_2 = units_2
        self.dense_units = dense_units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.clip_norm = clip_norm
        self.model = None
        self.scaler = None
        self.history = None

    def build_model(self, n_features: int):
        import tensorflow as tf
        from tensorflow import keras

        inputs = keras.Input(shape=(self.lookback, n_features))
        x = keras.layers.LSTM(
            self.units_1,
            return_sequences=True,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
        )(inputs)
        x = keras.layers.LSTM(
            self.units_2,
            return_sequences=False,
            dropout=self.dropout,
        )(x)
        x = keras.layers.Dense(self.dense_units, activation="relu")(x)
        outputs = keras.layers.Dense(1, activation="linear")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(clipnorm=self.clip_norm),
            loss=keras.losses.Huber(delta=1.0),
            metrics=["mae"],
        )
        return model

    def create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert 2D arrays to (samples, lookback, features) sequences."""
        Xs, ys = [], []
        for i in range(self.lookback, len(X)):
            Xs.append(X[i - self.lookback: i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.preprocessing import RobustScaler

        # Replace inf/nan with 0 before scaling
        X_train = np.where(np.isfinite(X_train), X_train, 0.0)
        X_val = np.where(np.isfinite(X_val), X_val, 0.0)

        # Scale features
        self.scaler = RobustScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        X_tr_seq, y_tr_seq = self.create_sequences(X_train_s, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val_s, y_val)

        n_features = X_tr_seq.shape[2]
        self.model = self.build_model(n_features)

        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=self.patience, restore_best_weights=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=7, monitor="val_loss", min_lr=1e-6
            ),
        ]

        self.history = self.model.fit(
            X_tr_seq, y_tr_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on raw (unscaled) feature array; returns predictions."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.where(np.isfinite(X), X, 0.0)
        X_s = self.scaler.transform(X)
        # Need at least lookback rows; return NaN for insufficient history
        if len(X_s) < self.lookback:
            return np.full(len(X_s), np.nan)
        X_seq = np.array([X_s[i - self.lookback: i] for i in range(self.lookback, len(X_s))])
        raw_preds = self.model.predict(X_seq, verbose=0).flatten()
        # Align with original index: first lookback rows get NaN
        result = np.full(len(X), np.nan)
        result[self.lookback:] = raw_preds
        return result

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        self.model.save(os.path.join(dir_path, "lstm_model.keras"))
        joblib.dump(self.scaler, os.path.join(dir_path, "scaler.joblib"))

    @classmethod
    def load(cls, dir_path: str) -> "LSTMForecaster":
        import tensorflow as tf
        obj = cls()
        obj.model = tf.keras.models.load_model(os.path.join(dir_path, "lstm_model.keras"))
        obj.scaler = joblib.load(os.path.join(dir_path, "scaler.joblib"))
        return obj
