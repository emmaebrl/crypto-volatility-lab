import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


class LSTMPipeline:
    def __init__(
        self,
        lookback: int = 25,
        forecast_horizon: int = 5,
        lstm_units: Tuple[int, int] = (48, 16),
        dropout_rate: float = 0.1,
        learning_rate: float = 0.01,
        epochs: int = 1,
        batch_size: int = 32,
        validation_split: float = 0.2,
        normalization: bool = True,
    ):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.normalization = normalization
        self.history = None
        self.model = None

    def create_lagged_features(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        assert X.shape[0] == y.shape[0]
        X = np.array(
            [
                X[t - self.lookback : t]
                for t in range(self.lookback, len(X) - self.forecast_horizon + 1)
            ]
        )
        y = np.array(
            [
                y[t : t + self.forecast_horizon]
                for t in range(self.lookback, len(y) - self.forecast_horizon + 1)
            ]
        )

        return X, y

    def normalize_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.normalization:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            y = scaler.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)
        return X, y

    def create_lstm_model(self, input_shape: Tuple[int, ...]) -> Sequential:
        model = Sequential(
            [
                Input(shape=input_shape),
                LSTM(
                    self.lstm_units[0],
                    activation="relu",
                    return_sequences=True,
                ),
                Dropout(self.dropout_rate),
                LSTM(self.lstm_units[1], activation="relu"),
                Dropout(self.dropout_rate),
                Dense(self.forecast_horizon),
            ]
        )
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse")
        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> Sequential:
        X, y = self.normalize_data(X, y)
        X, y = self.create_lagged_features(X, y)
        self.model = self.create_lstm_model(X.shape[1:])

        self.history = self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=1,
        )
        if self.history is not None:
            print(f"Train Loss (MSE from history): {self.history.history['loss'][-1]}")
            print(
                f"Validation Loss (MSE from history): {self.history.history['val_loss'][-1]}"
            )
        else:
            print("No training history available.")
        return self.model

    def get_history(self):
        if self.history:
            return self.history.history
        return None

    def evaluate_metrics(self, X: np.ndarray, y: np.ndarray):
        X, y = self.normalize_data(X, y)
        X, y = self.create_lagged_features(X, y)
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        else:
            predictions = self.model.predict(X, verbose=0)

            mse = mean_squared_error(y.flatten(), predictions.flatten())
            mae = mean_absolute_error(y.flatten(), predictions.flatten())
            mape = np.mean(np.abs((y - predictions) / y)) * 100

            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X, _ = self.normalize_data(X, np.zeros_like(X))
        X, _ = self.create_lagged_features(X, np.zeros_like(X))

        if self.model is None:
            raise ValueError("Model has not been trained yet")

        else:
            return self.model.predict(X)
