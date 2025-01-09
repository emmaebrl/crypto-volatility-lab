from arch import arch_model
import numpy as np
import pandas as pd


class FeaturesCreator:
    def compute_garch_forecast(self, log_returns: pd.Series):
        """Computes a GARCH(1,1) forecast for the given log returns."""
        print(log_returns.shape)
        log_returns = log_returns

        # Fit the GARCH(1,1) model
        model = arch_model(log_returns, vol="GARCH", p=1, q=1, rescale=False)
        res = model.fit(disp="off")

        # Get the conditional volatility (forecasted values)
        conditional_volatility = res.conditional_volatility

        return conditional_volatility

    def create_smoothed_volatility(self, volatility: pd.Series):
        """Creates a smoothed volatility time series using a rolling window."""
        weekly_window_size = 7  # weekly rolling window
        monthly_window_size = 30  # monthly rolling window
        volatility_weekly = volatility.rolling(window=weekly_window_size).mean()
        volatility_monthly = volatility.rolling(window=monthly_window_size).mean()

        return volatility_weekly, volatility_monthly

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
