import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioConstructor:
    def __init__(self, volatility_time_series: pd.DataFrame):
        """
        Initialise le constructeur de portefeuille Risk Parity.
        
        :param volatility_time_series: DataFrame contenant la volatilité historique ou prédite des actifs.
        """
        self.volatility_time_series = volatility_time_series

    def risk_parity_weights_simple(self) -> pd.DataFrame:
        """
        Calcule les poids du portefeuille Risk Parity en fonction de l'inverse de la volatilité.
        """
        inverse_volatility = 1 / self.volatility_time_series
        weights = inverse_volatility.div(inverse_volatility.sum(axis=1), axis=0)

        assert np.allclose(weights.sum(axis=1), 1), "Les poids ne sont pas normalisés à 1."
        return weights  
    
    def risk_parity_weights(self, cov_matrix: np.ndarray) -> pd.DataFrame:
    
        num_assets = cov_matrix.shape[0]

        def risk_budget_objective(weights):
            """Objectif : Équilibrer les contributions marginales au risque."""
            portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
            marginal_risk_contribution = (cov_matrix @ weights) / portfolio_volatility
            risk_contributions = weights * marginal_risk_contribution
            return np.std(risk_contributions)  # Minimise la dispersion des contributions au risque

    # Contraintes : somme des poids = 1 et poids positifs
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(num_assets)]
        initial_guess = np.ones(num_assets) / num_assets  # Poids égaux au départ

        result = minimize(
            risk_budget_objective, 
            initial_guess, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'disp': False}  # Mettre True pour voir les logs en cas de problème
    )

        if not result.success:
            raise RuntimeError("L'optimisation Risk Parity n'a pas convergé.")

    # 📌 Correction : S'assurer de retourner un VECTEUR et non une matrice carrée
        weights_optimized = pd.DataFrame([result.x], columns=self.volatility_time_series.columns)

        return weights_optimized
