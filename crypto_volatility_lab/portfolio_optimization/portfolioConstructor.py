
class PortfolioConstructor:
    def __init__(self) -> None:
        self.method = None
        self.data = None

    def return_weights(self) -> np.ndarray:
        if self.method == "equality":
            return self.equality()
        elif self.method == "risk-parity":
            return self.risk_parity()
        else:
            raise ValueError("Invalid method")
        