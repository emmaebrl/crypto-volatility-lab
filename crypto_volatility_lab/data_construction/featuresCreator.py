from arch import arch_model
import numpy as np
import pandas as pd


class FeaturesCreator:
    def __init__(
        self,
        data: pd.DataFrame,
        date_column_name: str = "Date",
        returns_column_name: str = "Log Returns",
        volatility_column_name: str = "Conditional Volatility",
    ):
        self.data = data.copy()
        self.date_column_name = date_column_name
        self.returns_column_name = returns_column_name
        self.volatility_column_name = volatility_column_name

    def compute_garch_forecast(self):
        """Computes a GARCH(1,1) forecast for the given log returns."""
        if self.returns_column_name not in self.data.columns:
            raise ValueError(f"Column '{self.returns_column_name}' not found in data.")

        log_returns = self.data[self.returns_column_name].dropna()

        if log_returns.empty:
            raise ValueError("Log returns series is empty after dropping NaN values.")

        # Fit the GARCH(1,1) model
        model = arch_model(log_returns, vol="GARCH", p=1, q=1, rescale=False)
        res = model.fit(disp="off")

        # Get the conditional volatility (forecasted values)
        conditional_volatility = res.conditional_volatility

        return conditional_volatility

    def create_smoothed_volatility(self, volatility: pd.Series):
        """Creates a smoothed volatility time series using a rolling window."""
        if volatility.empty:
            raise ValueError("Volatility series is empty.")

        weekly_window_size = 7  # Weekly rolling window
        monthly_window_size = 30  # Monthly rolling window

        volatility_weekly = volatility.rolling(
            window=weekly_window_size, min_periods=1
        ).mean()
        volatility_monthly = volatility.rolling(
            window=monthly_window_size, min_periods=1
        ).mean()

        return volatility_weekly, volatility_monthly

    def create_features(self):
        """Creates features for the LSTM model."""
        if self.volatility_column_name not in self.data.columns:
            raise ValueError(
                f"Column '{self.volatility_column_name}' not found in data."
            )

        # Extract volatility column
        volatility = self.data[self.volatility_column_name]

        # Compute the GARCH forecast
        try:
            volatility_garch = self.compute_garch_forecast()
        except Exception as e:
            raise RuntimeError(f"Error while computing GARCH forecast: {e}")

        # Create smoothed volatility features
        try:
            volatility_weekly, volatility_monthly = self.create_smoothed_volatility(
                volatility
            )
        except Exception as e:
            raise RuntimeError(
                f"Error while creating smoothed volatility features: {e}"
            )

        # Combine the features into a DataFrame
        features = pd.DataFrame(
            {
                "Date": self.data[self.date_column_name],
                "GARCH Volatility": volatility_garch,
                "Volatility (Weekly)": volatility_weekly,
                "Volatility (Monthly)": volatility_monthly,
                "Volatility": volatility,
            },
            index=self.data.index,
        )

        return features

    # def garch_volatility(self, log_returns: pd.Series):
    #     """Estimates parameters of a GARCH(1,1) model and returns the conditional volatility"""
    #     # Fit the GARCH(1,1) model
    #     model = arch_model(log_returns, vol="GARCH", p=1, q=1, rescale=False)
    #     res = model.fit(disp="off")

    #     omega = res.params["omega"]
    #     alpha = res.params["alpha[1]"]
    #     beta = res.params["beta[1]"]

    #     # Rolling forecast of the GARCH(1,1) model
    #     forecast_vol = np.sqrt(
    #         omega + alpha * res.resid**2 + beta * res.conditional_volatility**2
    #     )
    #     return forecast_vol
