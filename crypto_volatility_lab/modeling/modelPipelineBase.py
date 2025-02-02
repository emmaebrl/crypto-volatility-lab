from typing import Tuple, Optional
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from abc import ABC, abstractmethod
import pickle


class ModelPipelineBase(ABC):
    def __init__(
        self,
        lookback: int = 25,
        forecast_horizon: int = 5,
        learning_rate: float = 0.01,
        epochs: int = 1,
        batch_size: int = 32,
        validation_split: float = 0.2,
        normalize: bool = False,
        random_seed: int = 42,
    ):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.normalize = normalize
        self.random_seed = random_seed

        np.random.seed(self.random_seed)

        self.history = None
        self.model = None
        self.scaler_X = MinMaxScaler() if normalize else None
        self.scaler_y = MinMaxScaler() if normalize else None

    def create_lagged_features(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create lagged features for supervised learning."""
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

    @abstractmethod
    def create_model(self, input_shape: Tuple[int, ...]):
        """Abstract method for creating the machine learning model."""
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: int = 0):
        """Fit the model to the data."""
        if self.normalize and self.scaler_X is not None and self.scaler_y is not None:
            X = self.scaler_X.fit_transform(X)
            y = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        X, y = self.create_lagged_features(X, y)
        self.model = self.create_model(X.shape[1:])

        if self.model is None:
            raise ValueError("Model has not been created")
        self.history = self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=verbose,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if self.normalize and self.scaler_X is not None:
            X = self.scaler_X.transform(X)

        X, _ = self.create_lagged_features(X, np.zeros_like(X))

        if self.model is None:
            raise ValueError("Model has not been trained yet")

        predictions = self.model.predict(X)
        if self.normalize and self.scaler_y is not None:
            predictions = self.scaler_y.inverse_transform(predictions)

        return predictions

    def evaluate_metrics(self, X: np.ndarray, y: np.ndarray, only_last: bool = False):
        """Evaluate the GRU model using MSE, MAE, RMSE, and MAPE."""
        if self.normalize and self.scaler_X is not None and self.scaler_y is not None:
            X = self.scaler_X.transform(X)
            y = self.scaler_y.transform(y.reshape(-1, 1)).flatten()

        X, y = self.create_lagged_features(X, y)
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        predictions = self.model.predict(X)

        if self.normalize and self.scaler_y is not None:
            predictions = self.scaler_y.inverse_transform(predictions)
            y = self.scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()

        forecast_horizon = predictions.shape[1]
        y = y.reshape(-1, forecast_horizon)

        if only_last:  # Only evaluate the last forecast horizon
            y = y[:, -1]
            predictions = predictions[:, -1]
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)
            mape = np.mean(np.abs((y - predictions) / y)) * 100
            rmse = np.sqrt(mse)

            print("Evaluation Metrics for the Last Time Step:")
            print("-" * 60)
            print(f"{'Metric':<20}{'Value':<15}")
            print(f"{'MSE':<20}{mse:<15.4f}")
            print(f"{'RMSE':<20}{rmse:<15.4f}")
            print(f"{'MAE':<20}{mae:<15.4f}")
            print(f"{'MAPE (%)':<20}{mape:<15.4f}\n")

        else:
            mse_by_horizon = []
            mae_by_horizon = []
            mape_by_horizon = []
            rmse_by_horizon = []

            print("\nEvaluation Metrics by Time Step:")
            print(
                f"{'Time Step':<10}{'MSE':<15}{'RMSE':<15}{'MAE':<15}{'MAPE (%)':<15}"
            )
            print("-" * 60)

            for t in range(forecast_horizon):
                mse = mean_squared_error(y[:, t], predictions[:, t])
                mae = mean_absolute_error(y[:, t], predictions[:, t])
                mape = np.mean(np.abs((y[:, t] - predictions[:, t]) / y[:, t])) * 100
                rmse = np.sqrt(mse)
                mse_by_horizon.append(mse)
                mae_by_horizon.append(mae)
                mape_by_horizon.append(mape)
                rmse_by_horizon.append(rmse)

                print(f"{t+1:<10}{mse:<15.4f}{rmse:<15.4f}{mae:<15.4f}{mape:<15.4f}")

            overall_mse = mean_squared_error(y.flatten(), predictions.flatten())
            overall_mae = mean_absolute_error(y.flatten(), predictions.flatten())
            overall_mape = (
                np.mean(np.abs((y.flatten() - predictions.flatten()) / y.flatten()))
                * 100
            )
            overall_rmse = np.sqrt(overall_mse)

            print("\nOverall Evaluation Metrics:")
            print("-" * 60)
            print(f"{'Metric':<20}{'Value':<15}")
            print(f"{'Overall MSE':<20}{overall_mse:<15.4f}")
            print(f"{'Overall RMSE':<20}{overall_rmse:<15.4f}")
            print(f"{'Overall MAE':<20}{overall_mae:<15.4f}")
            print(f"{'Overall MAPE (%)':<20}{overall_mape:<15.4f}")

    def get_history(self) -> Optional[dict]:
        """Retrieve training history."""
        if self.history:
            return self.history.history
        return None

    def save(self, path: str):
        """Save the model and scalers to disk."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # dump self
        with open(path, "wb") as f:
            pickle.dump(self, f)
