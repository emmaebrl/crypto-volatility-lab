from typing import Optional
from arch import arch_model
import numpy as np
import pandas as pd


class TimeSeriesCreator:
    def __init__(
        self,
        data: pd.DataFrame,
        date_column_name: str = "Date",
        value_column_name: str = "Close",
    ):
        self.date_column_name = date_column_name
        self.value_column_name = value_column_name
        self.data = self._format_data(data)

    def _format_data(self, data: pd.DataFrame):
        """Formats the input data to ensure it contains the necessary columns."""
        if self.date_column_name not in data.columns:
            raise ValueError(
                f"Date column '{self.date_column_name}' not found in data."
            )
        if self.value_column_name not in data.columns:
            raise ValueError(
                f"Value column '{self.value_column_name}' not found in data."
            )
        data[self.date_column_name] = pd.to_datetime(data[self.date_column_name])
        data[self.value_column_name] = pd.to_numeric(
            data[self.value_column_name].str.replace(",", "")
        )

        # Sort the data by ascending date
        data.sort_values(by=self.date_column_name, inplace=True)
        return data

    def create_log_return_time_series(self):
        """Creates a time series of log returns from the input data."""
        log_returns = np.log(
            self.data[self.value_column_name]
            / self.data[self.value_column_name].shift(1)
        )  # log returns = log(Pt/Pt-1) = log(Pt) - log(Pt-1)
        log_returns = pd.Series(log_returns).fillna(0)
        return log_returns

    def create_volatility_time_series(self, window_size: int = 21):
        """Creates a time series of rolling volatility from the log return values."""
        log_returns = self.create_log_return_time_series()
        volatility = log_returns.rolling(window=window_size).std().fillna(0)
        return volatility

    def compute_garch_forecast(self, log_returns: pd.Series):
        """Computes a GARCH(1,1) forecast for the given log returns."""
        # Drop NaN values as GARCH requires complete data
        log_returns = log_returns

        # Fit the GARCH(1,1) model
        model = arch_model(log_returns, vol="GARCH", p=1, q=1, rescale=False)
        res = model.fit(disp="off")

        # Get the conditional volatility (forecasted values)
        conditional_volatility = res.conditional_volatility

        return conditional_volatility

    def create_time_series_for_NN_training(self, window_size: int = 21):
        """Creates a time series of log returns and volatility for training a neural network."""
        # Log returns and volatility with consistent indices
        log_returns = self.create_log_return_time_series()
        volatility = self.create_volatility_time_series(window_size=window_size)

        # Compute GARCH volatility
        garch_volatility = self.compute_garch_forecast(log_returns)

        # Rolling volatility averages
        weekly_window_size = 7  # weekly rolling window
        monthly_window_size = 30  # monthly rolling window
        volatility_weekly = volatility.rolling(window=weekly_window_size).mean()
        volatility_monthly = volatility.rolling(window=monthly_window_size).mean()

        # Align all features to the same index
        features = pd.DataFrame(
            {
                "Log Returns": log_returns,
                "Volatility": volatility,
                "GARCH Volatility": garch_volatility,
                "Volatility Weekly": volatility_weekly,
                "Volatility Monthly": volatility_monthly,
                "Date": self.data[self.date_column_name],
            }
        ).dropna()  # Drop rows with NaN values

        return features
