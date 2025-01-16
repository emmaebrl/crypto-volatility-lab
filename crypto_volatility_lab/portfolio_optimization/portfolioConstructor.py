import numpy as np


class PortfolioConstructor:
    def __init__(self) -> None:
        self.volatilities = None

    def risk_parity_weights(self, cov_matrix: np.ndarray) -> np.ndarray:
        inv_diag = np.diag(1 / np.sqrt(np.diag(cov_matrix)))
        weights = inv_diag @ cov_matrix @ inv_diag
        weights /= weights.sum()
        return weights

    def risk_parity_weights_simple(self, volatilities: np.ndarray) -> np.ndarray:
        inverse_volatilities = 1 / volatilities
        weights = inverse_volatilities / inverse_volatilities.sum()
        assert np.allclose(
            weights.sum(), 1
        ), "Weights do not sum to 1 for each time period."
        return weights

    def vol_target_weights(
        self, volatilities: np.ndarray, target_vol: float
    ) -> np.ndarray:
        weights = target_vol / volatilities
        weights /= weights.sum()
        assert np.allclose(
            weights.sum(), 1
        ), "Weights do not sum to 1 for each time period."
        return weights
