import numpy as np
import pandas as pd


class PortfolioConstructor:
    def __init__(self, volatility_time_series: pd.DataFrame):
        self.volatility_time_series = volatility_time_series

    def risk_parity_weights_simple(self) -> np.ndarray:
        inverse_volatility = 1 / self.volatility_time_series
        weights = inverse_volatility.div(inverse_volatility.sum(axis=1), axis=0)
        assert np.allclose(
            weights.sum(axis=1), 1
        ), "Weights do not sum to 1 for each time period."
        return weights.values
    

    def risk_parity_weights(self, cov_matrix: np.ndarray) -> np.ndarray:
        inv_diag = np.diag(1 / np.sqrt(np.diag(cov_matrix)))
        weights = inv_diag @ cov_matrix @ inv_diag
        weights /= weights.sum()
        return weights
