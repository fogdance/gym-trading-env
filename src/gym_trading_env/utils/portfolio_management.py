# src/gym_trading_env/utils/portfolio_management.py

from decimal import Decimal

class Portfolio:
    def __init__(self, asset: Decimal = Decimal('0.0'), fiat: Decimal = Decimal('0.0')):
        self.asset = asset
        self.fiat = fiat

    def update(self, asset: Decimal, fiat: Decimal):
        self.asset = asset
        self.fiat = fiat

    def valorisation(self, price: Decimal) -> Decimal:
        """
        Calculates the total portfolio value.

        Args:
            price (Decimal): Current price of the asset.

        Returns:
            Decimal: Total portfolio value.
        """
        return self.fiat + (self.asset * price)

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
