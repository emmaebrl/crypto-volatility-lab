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
        self.data = self._check_data(data)

    def _check_data(self, data: pd.DataFrame):
        """Formats the input data to ensure it contains the necessary columns."""
        if self.date_column_name not in data.columns:
            raise ValueError(
                f"Date column '{self.date_column_name}' not found in data."
            )
        if self.value_column_name not in data.columns:
            raise ValueError(
                f"Value column '{self.value_column_name}' not found in data."
            )
        # Sort the data by ascending date
        data.sort_values(by=self.date_column_name, inplace=True)
        return data

    def create_log_return_time_series(self):
        """Creates a time series of log returns from the input data."""
        log_returns = np.log(
            self.data[self.value_column_name]
            / self.data[self.value_column_name].shift(1)
        )
        return pd.Series(log_returns)

    def create_volatility_time_series(self, window_size: int = 30):
        """Creates a time series of rolling volatility from the log return values."""
        log_returns = self.create_log_return_time_series()
        volatility = log_returns.rolling(window=window_size).std()
        return volatility

    def create_time_series(self, window_size: int = 30):
        """Creates a time series of log returns and volatility for training a neural network."""
        log_returns = self.create_log_return_time_series()
        volatility = self.create_volatility_time_series(window_size)
        return pd.DataFrame(
            {
                "Log Returns": log_returns,
                "Volatility": volatility,
                "Date": self.data[self.date_column_name],
            }
        ).dropna()
