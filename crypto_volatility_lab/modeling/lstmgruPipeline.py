from typing import Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
        normalize: bool = False,
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
        self.normalize = normalize

        self.history = None
        self.model = None
        self.scaler_X = MinMaxScaler() if normalize else None
        self.scaler_y = MinMaxScaler() if normalize else None

    def create_lagged_features(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        assert X.shape[0] == y.shape[0]

        if self.normalize:
            if self.scaler_X is not None:
                X = self.scaler_X.fit_transform(X)
            if self.scaler_y is not None:
                y = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

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
                Dense(self.forecast_horizon),
            ]
        )
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse")
        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> Sequential:
        X, y = self.create_lagged_features(X, y)
        self.model = self.create_mixed_model(X.shape[1:])

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
        X, y = self.create_lagged_features(X, y)
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        else:
            predictions = self.model.predict(X, verbose=0)

            if self.normalize and self.scaler_y is not None:
                predictions = self.scaler_y.inverse_transform(predictions)
                y = self.scaler_y.inverse_transform(y)

            mse = mean_squared_error(y.flatten(), predictions.flatten())
            mae = mean_absolute_error(y.flatten(), predictions.flatten())
            mape = np.mean(np.abs((y - predictions) / y)) * 100

            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X, _ = self.create_lagged_features(X, np.zeros_like(X))

        if self.model is None:
            raise ValueError("Model has not been trained yet")

        else:
            predictions = self.model.predict(X)
            if self.normalize and self.scaler_y is not None:
                predictions = self.scaler_y.inverse_transform(predictions)
                print("Predictions have been inverse transformed")

        return predictions
