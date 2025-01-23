from typing import Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import GRU, Dense, Dropout, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
import tensorflow as tf
import os

os.environ["TF_DETERMINISTIC_OPS"] = "1"


class GRUPipeline:
    def __init__(
        self,
        lookback: int = 25,
        forecast_horizon: int = 5,
        gru_units: Tuple[int, int] = (48, 16),
        dropout_rate: float = 0.1,
        learning_rate: float = 0.01,
        epochs: int = 1,
        batch_size: int = 32,
        validation_split: float = 0.2,
        normalize: bool = False,
        random_seed: int = 42,
    ):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.normalize = normalize
        self.random_seed = random_seed

        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        self.history = None
        self.model = None
        self.scaler_X = MinMaxScaler() if normalize else None
        self.scaler_y = MinMaxScaler() if normalize else None

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

    def create_gru_model(self, input_shape: Tuple[int, ...]) -> Sequential:
        model = Sequential(
            [
                Input(shape=input_shape),
                GRU(
                    self.gru_units[0],
                    activation="relu",
                    return_sequences=True,
                ),
                Dropout(self.dropout_rate),
                GRU(self.gru_units[1], activation="relu"),  # relu ?
                Dropout(self.dropout_rate),
                Dense(self.forecast_horizon),
            ]
        )
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse")
        return model

    def get_history(self):
        if self.history:
            return self.history.history
        return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Sequential:
        if self.normalize and self.scaler_X is not None and self.scaler_y is not None:
            X = self.scaler_X.fit_transform(X)
            y = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        X, y = self.create_lagged_features(X, y)
        self.model = self.create_gru_model(X.shape[1:])

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

    def evaluate_metrics(self, X: np.ndarray, y: np.ndarray):
        if self.normalize and self.scaler_X is not None and self.scaler_y is not None:
            X = self.scaler_X.transform(X)
            y = self.scaler_y.transform(y.reshape(-1, 1)).flatten()

            X, y = self.create_lagged_features(X, y)
            if self.model is None:
                raise ValueError("Model has not been trained yet")
            else:
                predictions = self.model.predict(X, verbose=0)

                if self.normalize and self.scaler_y is not None:
                    predictions = self.scaler_y.inverse_transform(predictions)
                    y = self.scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()

                forecast_horizon = predictions.shape[1]
                y = y.reshape(-1, forecast_horizon)
                mse_by_horizon = []
                mae_by_horizon = []
                mape_by_horizon = []

                print("\nEvaluation Metrics by Time Step:")
                print(f"{'Time Step':<10}{'MSE':<15}{'MAE':<15}{'MAPE (%)':<15}")
                print("-" * 45)

                for t in range(forecast_horizon):
                    mse = mean_squared_error(y[:, t], predictions[:, t])
                    mae = mean_absolute_error(y[:, t], predictions[:, t])
                    mape = (
                        np.mean(np.abs((y[:, t] - predictions[:, t]) / y[:, t])) * 100
                    )
                    mse_by_horizon.append(mse)
                    mae_by_horizon.append(mae)
                    mape_by_horizon.append(mape)

                    print(f"{t+1:<10}{mse:<15.4f}{mae:<15.4f}{mape:<15.4f}")

                overall_mse = mean_squared_error(y.flatten(), predictions.flatten())
                overall_mae = mean_absolute_error(y.flatten(), predictions.flatten())
                overall_mape = (
                    np.mean(np.abs((y.flatten() - predictions.flatten()) / y.flatten()))
                    * 100
                )

                print("\nOverall Evaluation Metrics:")
                print("-" * 45)
                print(f"{'Metric':<20}{'Value':<15}")
                print(f"{'Overall MSE':<20}{overall_mse:<15.4f}")
                print(f"{'Overall MAE':<20}{overall_mae:<15.4f}")
                print(f"{'Overall MAPE (%)':<20}{overall_mape:<15.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.normalize and self.scaler_X is not None:
            X = self.scaler_X.transform(X)

        X, _ = self.create_lagged_features(X, np.zeros_like(X))

        if self.model is None:
            raise ValueError("Model has not been trained yet")

        else:
            predictions = self.model.predict(X)
            if self.normalize and self.scaler_y is not None:
                predictions = self.scaler_y.inverse_transform(predictions)

        return predictions
