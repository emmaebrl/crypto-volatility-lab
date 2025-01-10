import numpy as np


import numpy as np


class RiskParityPortfolio:
    def __init__(self):
        """
        Initialiser la classe RiskParityPortfolio.
        """
        self.weights = None

    def compute_weights(self, asset_std_devs):
        """
        Calcule les poids pour un portefeuille à parité de risque.

        Paramètres :
        - asset_std_devs (array-like) : Une liste ou un tableau contenant les écarts-types des actifs.

        Retourne :
        - weights (numpy.ndarray) : Les poids calculés pour chaque actif.
        """
        asset_std_devs = np.array(asset_std_devs)

        if np.any(asset_std_devs <= 0):
            raise ValueError("Les écarts-types doivent être strictement positifs.")

        inverse_std_devs = 1 / asset_std_devs
        total_inverse_std_devs = np.sum(inverse_std_devs)

        self.weights = inverse_std_devs / total_inverse_std_devs
        return self.weights

    def get_weights(self):
        """
        Retourne les poids calculés.

        Retourne :
        - weights (numpy.ndarray ou None) : Les poids calculés, ou None si les poids n'ont pas encore été calculés.
        """
        if self.weights is None:
            raise ValueError(
                "Les poids n'ont pas encore été calculés. Appelez compute_weights()."
            )
        return self.weights
