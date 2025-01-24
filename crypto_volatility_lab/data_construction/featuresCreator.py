import pandas as pd
import numpy as np
from arch import arch_model


class FeaturesCreator:
    def __init__(
        self,
        data: pd.DataFrame,
        log_returns_column_name: str = "Log Returns",
        volatility_column_name: str = "Volatility",
        high_column_name: str = "High",
        low_column_name: str = "Low",
        volume_column_name: str = "Volume",
    ) -> None:
        """
        Initializes the FeaturesCreator object with a time series.

        Args:
            volatility (pd.Series): Time series of volatility.
        """
        self.data = data
        self.volatility_column_name = volatility_column_name
        self.log_returns_column_name = log_returns_column_name
        self.high_column_name = high_column_name
        self.low_column_name = low_column_name
        self.volume_column_name = volume_column_name
        self.transformed_data = data.copy()
        self.features_names = []

    def create_smoothed_volatility(self):
        """
        Creates and adds smoothed volatility series (weekly and monthly) to the features DataFrame.
        """
        volatility = self.data[self.volatility_column_name]
        weekly_window_size = 7  # Weekly rolling window
        monthly_window_size = 30  # Monthly rolling window
        self.transformed_data["Weekly Volatility"] = volatility.rolling(
            window=weekly_window_size, min_periods=1
        ).mean()
        self.transformed_data["Monthly Volatility"] = volatility.rolling(
            window=monthly_window_size, min_periods=1
        ).mean()

    def create_garch_volatility(self):
        """Estimates parameters of a GARCH(1,1) model and returns the conditional volatility"""
        log_returns = self.data[self.log_returns_column_name].dropna()
        am = arch_model(log_returns, vol="GARCH", p=1, q=1, rescale=False)
        res = am.fit(disp="off")

        omega = res.params["omega"]
        alpha = res.params["alpha[1]"]
        beta = res.params["beta[1]"]

        forecast_vol = np.sqrt(
            omega + alpha * res.resid**2 + beta * res.conditional_volatility**2
        )
        self.transformed_data["GARCH Volatility"] = forecast_vol

    def create_log_trading_range(self):
        """
        Creates and adds log trading range series to the features DataFrame.
        """
        high = self.data[self.high_column_name]
        low = self.data[self.low_column_name]
        self.transformed_data["Log Trading Range"] = np.log(high) - np.log(low)

    def create_log_volume_change(self):
        """
        Creates and adds log volume change series to the features DataFrame.
        """
        volume = self.data[self.volume_column_name]
        self.transformed_data["Log Volume Change"] = np.log(volume) - np.log(
            volume.shift(1)
        )

    def create_all_features(self):
        """
        Creates all features and adds them to the features DataFrame.
        """
        self.create_smoothed_volatility()
        self.create_garch_volatility()
        self.create_log_trading_range()
        self.create_log_volume_change()
        self.features_names = [
            "Volatility",
            "Weekly Volatility",
            "Monthly Volatility",
            "GARCH Volatility",
            "Log Trading Range",
            "Log Volume Change",
        ]
