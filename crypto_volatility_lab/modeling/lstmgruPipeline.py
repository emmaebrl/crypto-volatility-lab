from typing import Tuple
import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from sklearn.preprocessing import MinMaxScaler


class LSTMGRUPipeline:
    def __init__(
        self,
        lookback: int = 25,
        forecast_horizon: int = 5,
        lstm_unit: int = 48,
        gru_unit: int = 16,
        dropout_rate: float = 0.1,
        learning_rate: float = 0.01,
        epochs: int = 1,
        batch_size: int = 32,
        validation_split: float = 0.2,
        scale_data: bool = False,
    ):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.lstm_unit = lstm_unit
        self.gru_unit = gru_unit
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.scale_data = scale_data
        self.scaler = MinMaxScaler() if scale_data else None
        self.history = None
        self.model = None

    def create_lagged_features(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array(
            [
                data[t - self.lookback : t]
                for t in range(self.lookback, len(data) - self.forecast_horizon + 1)
            ]
        )
        y = np.array(
            [
                data[t : t + self.forecast_horizon]
                for t in range(self.lookback, len(data) - self.forecast_horizon + 1)
            ]
        )
        return X, y

    def scale_features(self, X: np.ndarray) -> np.ndarray:
        if self.scaler:
            n_samples, n_timesteps, n_features = X.shape
            X = self.scaler.fit_transform(X.reshape(-1, n_features)).reshape(X.shape)
        return X

    def create_mixed_model(self, input_shape: Tuple[int, ...]) -> Sequential:
        model = Sequential(
            [
                Input(shape=input_shape),
                LSTM(
                    self.lstm_unit,
                    activation="relu",
                    return_sequences=True,
                ),
                Dropout(self.dropout_rate),
                GRU(self.gru_unit, activation="relu"),
                Dropout(self.dropout_rate),
                Dense(1),
            ]
        )
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse")
        return model

    def run(self, data: np.ndarray) -> Sequential:
        X, y = self.create_lagged_features(data)

        if self.scale_data:
            X = self.scale_features(X)

        self.model = self.create_mixed_model(X.shape[1:])

        self.history = self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=1,
        )

        return self.model

    def get_history(self):
        if self.history:
            return self.history.history
        return None

    def predict(self, new_data: np.ndarray) -> np.ndarray:
        X_new, _ = self.create_lagged_features(new_data)
        if self.scale_data and self.scaler:
            X_new = self.scaler.transform(X_new.reshape(-1, X_new.shape[2])).reshape(
                X_new.shape
            )
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        else:
            return self.model.predict(X_new)
