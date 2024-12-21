# src/gym_trading_env/utils/portfolio_management.py

class Portfolio:
    """
    A class to manage the portfolio, including assets and fiat balance.
    """
    
    def __init__(self, asset: float = 0.0, fiat: float = 0.0):
        self.asset = asset
        self.fiat = fiat
    
    def valorisation(self, price: float) -> float:
        """
        Calculates the total portfolio value.
        
        Args:
            price (float): Current price of the asset.
        
        Returns:
            float: Total portfolio value.
        """
        return self.asset * price + self.fiat
    
    def update(self, asset: float = None, fiat: float = None):
        """
        Updates the portfolio's assets and fiat balance.
        
        Args:
            asset (float, optional): New asset value.
            fiat (float, optional): New fiat balance.
        """
        if asset is not None:
            self.asset = asset
        if fiat is not None:
            self.fiat = fiat
    
    def get_distribution(self, price: float) -> dict:
        """
        Gets the distribution of assets and fiat in the portfolio.
        
        Args:
            price (float): Current price of the asset.
        
        Returns:
            dict: Distribution of assets and fiat.
        """
        total = self.valorisation(price)
        return {
            "asset": max(0, self.asset),
            "fiat": max(0, self.fiat),
            "borrowed_asset": max(0, -self.asset),
            "borrowed_fiat": max(0, -self.fiat),
        }
