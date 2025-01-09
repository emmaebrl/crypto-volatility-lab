from typing import Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import GRU, Dense, Dropout, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


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
        scale_data: bool = False,
    ):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.scale_data = scale_data
        self.scaler = MinMaxScaler() if scale_data else None
        self.history = None
        self.model = None

    def create_lagged_features(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def scale_features(self, X: np.ndarray) -> np.ndarray:
        if self.scaler:
            n_samples, n_timesteps, n_features = X.shape
            X = self.scaler.fit_transform(X.reshape(-1, n_features)).reshape(X.shape)
        return X

    def scale_targets(self, y: np.ndarray) -> np.ndarray:
        if self.scaler:
            y = self.scaler.fit_transform(y.reshape(-1, 1)).reshape(y.shape)
        return y

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> Sequential:
        X, y = self.create_lagged_features(X, y)

        if self.scale_data:
            X = self.scale_features(X)
            y = self.scale_targets(y)

        self.model = self.create_gru_model(X.shape[1:])

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

    def evaluate_metrics(self, X: np.ndarray, y: np.ndarray):
        X, y = self.create_lagged_features(X, y)

        if self.scale_data:
            X = self.scale_features(X)
            y = self.scale_targets(y)

        if self.model is None:
            raise ValueError("Model has not been trained yet")

        else:
            predictions = self.model.predict(X, verbose=0)

            mse = mean_squared_error(y.flatten(), predictions.flatten())
            mae = mean_absolute_error(y.flatten(), predictions.flatten())
            mape = np.mean(np.abs((y - predictions) / y)) * 100

            if self.history is not None:
                print(
                    f"Train Loss (MSE from history): {self.history.history['loss'][-1]}"
                )
                print(
                    f"Validation Loss (MSE from history): {self.history.history['val_loss'][-1]}"
                )
            else:
                print("No training history available.")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X, _ = self.create_lagged_features(X, np.zeros_like(X))

        if self.scale_data and self.scaler:
            X = self.scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

        if self.model is None:
            raise ValueError("Model has not been trained yet")

        else:
            return self.model.predict(X)

    def plot_predictions(self, X: np.ndarray, y: np.ndarray):

        X, y = self.create_lagged_features(X, y)

        if self.scale_data:
            X = self.scale_features(X)
            y = self.scale_targets(y)

        if self.model is None:
            raise ValueError("Model has not been trained yet")

        predictions = self.model.predict(X, verbose=0)

        plt.figure(figsize=(14, 7))
        plt.plot(y.flatten(), label="Actual Values")
        plt.plot(predictions.flatten(), label="Predicted Values")
        plt.title("Predictions vs Actual Values")
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.legend()
        plt.show()
