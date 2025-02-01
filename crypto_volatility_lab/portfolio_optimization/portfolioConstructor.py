from typing import Optional
import numpy as np
import pandas as pd


class PortfolioConstructor:
    def __init__(self, volatility_time_series: pd.DataFrame):
        self.volatility_time_series = volatility_time_series

    def risk_parity_weights_simple(self) -> pd.DataFrame:
        """
        Calcule les poids de Risk Parity basés sur l'inverse des volatilités.
        """
        inverse_volatility = 1 / (self.volatility_time_series)
        weights = inverse_volatility.div(inverse_volatility.sum(axis=1), axis=0)

        assert np.allclose(
            weights.sum(axis=1), 1
        ), "❌ ERREUR: La somme des poids doit être égale à 1."
        return weights

    def vol_target_weights(self, target_volatility: float = 0.15) -> pd.DataFrame:
        print(self.volatility_time_series.describe())

        weights = target_volatility / self.volatility_time_series
        print(weights)

        # weights = weights.clip(lower=0.1, upper=w_max)
        weights = weights.div(weights.sum(axis=1), axis=0)

        return weights
