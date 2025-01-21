import pandas as pd
import numpy as np
from arch import arch_model


class FeaturesCreator:
    def __init__(
        self,
        data: pd.DataFrame,
        log_returns_column_name: str,
        volatility_column_name: str,
    ) -> None:
        """
        Initializes the FeaturesCreator object with a time series.

        Args:
            volatility (pd.Series): Time series of volatility.
        """
        self.data = data
        self.volatility_column_name = volatility_column_name
        self.log_returns_column_name = log_returns_column_name
        self.transformed_data = data.copy()
        self.features_names = []

    def create_smoothed_volatility(self):
        """
        Creates and adds smoothed volatility series (weekly and monthly) to the features DataFrame.

        Adds two columns to the self.features DataFrame:
            - "volatility_weekly_smoothed"
            - "volatility_monthly_smoothed"
        """
        volatility = self.data[self.volatility_column_name]

        weekly_window_size = 7  # Weekly rolling window
        monthly_window_size = 30  # Monthly rolling window

        self.transformed_data["volatility_weekly_smoothed"] = volatility.rolling(
            window=weekly_window_size, min_periods=1
        ).mean()
        self.transformed_data["volatility_monthly_smoothed"] = volatility.rolling(
            window=monthly_window_size, min_periods=1
        ).mean()

    def create_RSI_feature(self, window_size: int = 14):
        """
        Creates and adds the Relative Strength Index (RSI) feature to the features DataFrame.
        """
        # Compute the daily returns
        daily_returns = self.data[self.log_returns_column_name]

        # Compute the difference between the daily returns
        delta = daily_returns.diff()

        # Get the positive and negative gains
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Compute the average gain and loss
        avg_gain = gain.rolling(window=window_size).mean()
        avg_loss = loss.rolling(window=window_size).mean()

        # Compute the Relative Strength
        RS = avg_gain / avg_loss

        # Compute the Relative Strength Index
        RSI = 100 - (100 / (1 + RS))

        self.transformed_data["RSI"] = RSI

    def create_all_features(self):
        """
        Creates all features and adds them to the features DataFrame.
        """
        self.create_smoothed_volatility()
        self.create_RSI_feature()
        self.features_names = [
            "Volatility",
            "volatility_weekly_smoothed",
            "volatility_monthly_smoothed",
            "RSI",
        ]

    def garch_volatility(self, log_returns: pd.Series):
        """Estimates parameters of a GARCH(1,1) model and returns the conditional volatility"""
        # Fit the GARCH(1,1) model
        model = arch_model(log_returns, vol="GARCH", p=1, q=1, rescale=False)
        res = model.fit(disp="off")

        omega = res.params["omega"]
        alpha = res.params["alpha[1]"]
        beta = res.params["beta[1]"]

        forecast_vol = np.sqrt(
            omega + alpha * res.resid**2 + beta * res.conditional_volatility**2
        )
        return forecast_vol
