import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioConstructor:

    def __init__(self, volatility_time_series: pd.DataFrame, target_vol_factor=1.0):
        """
        Initialise le constructeur de portefeuille Risk Parity avec volatilité cible basée sur le marché.

        :param volatility_time_series: DataFrame contenant la volatilité prédite des actifs.
        :param target_vol_factor: Facteur d'ajustement pour la volatilité cible (ex: 1.0 = 100%, 0.8 = 80%).
        """
        self.volatility_time_series = volatility_time_series
        self.target_vol_factor = target_vol_factor
        self.target_volatility = self.calculate_target_volatility()
    
    def calculate_target_volatility(self):
        """
        Calcule la volatilité cible basée sur la dernière volatilité des actifs du marché.
        """
        latest_volatility_pred = self.volatility_time_series.iloc[-1]  # Dernière période
        return self.target_vol_factor * latest_volatility_pred.mean()


    def risk_parity_weights_simple(self) -> pd.DataFrame:
        """
        Calcule les poids du portefeuille Risk Parity en fonction de l'inverse de la volatilité.
        """
        inverse_volatility = 1 / self.volatility_time_series
        weights = inverse_volatility.div(inverse_volatility.sum(axis=1), axis=0)

        assert np.allclose(weights.sum(axis=1), 1), "Les poids ne sont pas normalisés à 1."
        return weights  
    
    def risk_parity_weights_simple_target(self) -> pd.DataFrame:
        """
        Calcule les poids du portefeuille Risk Parity simple et les ajuste pour atteindre `target_volatility`.
        """
        weights = self.risk_parity_weights_simple()
        portfolio_volatility = (weights * self.volatility_time_series).sum(axis=1)
        adjustment_factor = self.target_volatility / portfolio_volatility
        weights_adjusted = weights.mul(adjustment_factor, axis=0)
        weights_adjusted = weights_adjusted.div(weights_adjusted.sum(axis=1), axis=0)

        return weights_adjusted
    


    
    
    def risk_parity_weights(self, cov_matrix: np.ndarray) -> pd.DataFrame:
    
        num_assets = cov_matrix.shape[0]

        def risk_budget_objective(weights):
            """Objectif : Équilibrer les contributions marginales au risque."""
            weights = np.array(weights).reshape(-1, 1)  # 📌 Ajouté : S'assurer que weights est une colonne
            portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights).flatten()[0]  # Convertir en scalaire
            marginal_risk_contribution = (cov_matrix @ weights).flatten() / portfolio_volatility
            risk_contributions = (weights.flatten() * marginal_risk_contribution)
            return np.std(risk_contributions)  # Minimise la dispersion des contributions au risque

    # 📌 Contraintes : somme des poids = 1 et poids positifs
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(num_assets)]
        initial_guess = np.ones(num_assets) / num_assets  # Poids égaux au départ

        result = minimize(
            risk_budget_objective, 
            initial_guess, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'disp': False}  
        )

        if not result.success:
            raise RuntimeError("L'optimisation Risk Parity n'a pas convergé.")

    # 📌 Correction : Retourner un DataFrame contenant un VECTEUR de poids (1,3) et non une matrice (3x3)
        weights_optimized = pd.DataFrame([result.x], columns=self.volatility_time_series.columns)

        return weights_optimized
    
    