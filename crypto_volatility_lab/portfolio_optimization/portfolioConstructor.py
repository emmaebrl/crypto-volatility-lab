from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioConstructor:
    def __init__(
        self,
        volatility_time_series: pd.DataFrame,
        target_vol_factor: Optional[float] = 1.0,
    ):
        """Initializes the PortfolioConstructor object with a time series of asset volatilities."""
        self.volatility_time_series = volatility_time_series
        self.target_vol_factor = target_vol_factor

    def risk_parity_weights_simple(self) -> pd.DataFrame:
        """
        Calculates the simple Risk Parity weights based on the inverse of asset volatilities.
        """
        inverse_volatility = 1 / self.volatility_time_series
        weights = inverse_volatility.div(inverse_volatility.sum(axis=1), axis=0)

        assert np.allclose(
            weights.sum(axis=1), 1
        ), "The sum of weights for each period should be equal to 1."
        return weights

    def calculate_target_volatility(self):
        """
    Calculates the target volatility based on the latest volatility prediction.
    """
        latest_volatility_pred = self.volatility_time_series.iloc[-1]
        if self.target_vol_factor:
            target_volatility = self.target_vol_factor * latest_volatility_pred.median()  # 🔥 Changer mean() en median()
        else:
            target_volatility = latest_volatility_pred.median()
        return target_volatility

    def risk_parity_weights_simple_target(self) -> pd.DataFrame:
        """
        Calculates the Risk Parity weights based on the inverse of asset volatilities.
        """
        target_volatility = self.calculate_target_volatility()
        weights = self.risk_parity_weights_simple()
        portfolio_volatility = (weights * self.volatility_time_series).sum(axis=1)
        adjustment_factor = (target_volatility / portfolio_volatility).clip(lower=0.5, upper=2.0)

        weights_adjusted = weights.mul(adjustment_factor, axis=0)
        weights_adjusted = weights_adjusted.div(weights_adjusted.sum(axis=1), axis=0)

        return weights_adjusted

    # def _risk_budget_objective(
    #     self, weights: np.ndarray, cov_matrix: np.ndarray
    # ) -> float:
    #     """Objective function to minimize for the risk budget optimization."""
    #     weights = np.array(weights).reshape(-1, 1)
    #     portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights).flatten()[0]
    #     marginal_risk_contribution = (
    #         cov_matrix @ weights
    #     ).flatten() / portfolio_volatility
    #     risk_contributions = weights.flatten() * marginal_risk_contribution
    #     return float(np.std(risk_contributions))

    # def risk_parity_weights(self) -> pd.DataFrame:
    #     # compute cov_matrix for each time step
    #     cov_matrix = self.volatility_time_series.cov().values
    #     num_assets = cov_matrix.shape[0]
    #     constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    #     bounds = [(0, 1) for _ in range(num_assets)]
    #     initial_guess = np.ones(num_assets) / num_assets

    #     result = minimize(
    #         self._risk_budget_objective,
    #         initial_guess,
    #         method="SLSQP",
    #         bounds=bounds,
    #         constraints=constraints,
    #         options={"disp": False},
    #     )

    #     if not result.success:
    #         raise RuntimeError("L'optimisation Risk Parity n'a pas convergé.")
    #     weights_optimized = pd.DataFrame(
    #         [result.x], columns=self.volatility_time_series.columns
    #     )

    #     return weights_optimized
