from typing import Optional
import numpy as np
import pandas as pd


class PortfolioConstructor:
    def __init__(
        self,
        volatility_time_series: pd.DataFrame,
        target_vol_factor: Optional[float] = 1.0,
    ):
        """Initialise le PortfolioConstructor avec une série temporelle de volatilités."""
        self.volatility_time_series = volatility_time_series

        # ✅ Vérification et conversion
        if target_vol_factor is None:
            raise ValueError("❌ ERREUR: `target_vol_factor` ne peut pas être None !")
        self.target_vol_factor = float(target_vol_factor)

        # ✅ Calcul automatique de target_volatility avec la NOUVELLE APPROCHE
        self.target_volatility = self.calculate_target_volatility()

    def risk_parity_weights_simple(self) -> pd.DataFrame:
        """
        Calcule les poids de Risk Parity basés sur l'inverse des volatilités.
        """
        inverse_volatility = 1 / (self.volatility_time_series + 1e-8)  # ✅ Évite la division par zéro
        weights = inverse_volatility.div(inverse_volatility.sum(axis=1), axis=0)

        assert np.allclose(
            weights.sum(axis=1), 1
        ), "❌ ERREUR: La somme des poids doit être égale à 1."
        return weights

    def calculate_target_volatility(self) -> float:
        """
        Calcule une volatilité cible dynamique basée sur les prédictions LSTM.
        """
        vol_series = self.volatility_time_series.mean(axis=1)  # Moyenne des volatilités sur le temps

        # ✅ Nouvelle méthode : Accélération de la volatilité avec `diff().diff()`
        vol_acceleration = vol_series.diff().diff().fillna(0)  

        # ✅ Transformation exponentielle pour amplifier les différences
        dynamic_factor = np.exp(np.clip(vol_acceleration.iloc[-1], -0.2, 0.2))  

        # ✅ Calcul d’une volatilité cible amplifiée
        base_vol = vol_series.iloc[-1]  
        target_volatility = self.target_vol_factor * base_vol * dynamic_factor

        return float(target_volatility)

    def risk_parity_weights_simple_target(self) -> pd.DataFrame:
        """
        Calcule les poids de Risk Parity ajustés pour une volatilité cible dynamique.
        """
        target_volatility = self.target_volatility
        weights = self.risk_parity_weights_simple()
        portfolio_volatility = (weights * self.volatility_time_series).sum(axis=1)

        # ✅ Éviter la division par zéro
        portfolio_volatility = portfolio_volatility.replace(0, np.nan).fillna(1e-8)

        # ✅ Transformation exponentielle pour rendre l’effet plus visible
        adjustment_factor = np.exp((target_volatility / portfolio_volatility) - 1)  

        # ✅ Nouvelle plage pour maximiser l'effet
        adjustment_factor = adjustment_factor.clip(lower=0.7, upper=1.5)  

        # ✅ Application de l'ajustement aux poids
        weights_adjusted = weights.mul(adjustment_factor, axis=0)
        weights_adjusted = weights_adjusted.div(weights_adjusted.sum(axis=1), axis=0)

        return weights_adjusted
