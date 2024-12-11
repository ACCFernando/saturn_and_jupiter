class PaperTrader:
    """
    Simulates trade executions and PnL calculation without risking real capital.

    Attributes
    ----------
    balance : float
        The simulated USD balance.
    position : dict or None
        Current open position. None if no open positions.
        Example:
        {
          "side": "BUY",  # or "SELL"
          "amount": 100.0,
          "entry_price": 30000.0
        }
    """

    def __init__(self, starting_balance: float):
        """
        Initialize the PaperTrader with a starting balance.

        Parameters
        ----------
        starting_balance : float
            Initial simulated capital in USD.
        """
        self.balance = starting_balance
        self.position = None

    def open_position(self, side: str, amount: float, price: float):
        """
        Open a new position.

        Parameters
        ----------
        side : str
            "BUY" or "SELL"
        amount : float
            Amount in USD allocated to the position.
        price : float
            The entry price of the asset.

        Raises
        ------
        Exception
            If attempting to open a new position when one is already open.
        """
        if self.position is not None:
            raise Exception("Position already open!")
        self.position = {
            "side": side,
            "amount": amount,
            "entry_price": price
        }

    def close_position(self, price: float) -> float:
        """
        Close the current position and realize profit or loss.
        
        Parameters
        ----------
        price : float
            The exit price of the asset at the time of closing.

        Returns
        -------
        float
            The profit or loss (PnL) realized on the trade.
        """
        if self.position is None:
            return 0.0

        side = self.position["side"]
        amount = self.position["amount"]
        entry_price = self.position["entry_price"]
        quantity = amount / entry_price  # how many units of crypto were effectively bought or sold

        if side == "BUY":
            # PnL = (exit - entry) * quantity
            pnl = (price - entry_price) * quantity
        else:
            # SELL (short): PnL = (entry - exit) * quantity
            pnl = (entry_price - price) * quantity

        self.balance += pnl
        self.position = None
        return pnl

    def get_balance(self) -> float:
        """
        Get the current simulated balance.

        Returns
        -------
        float
            Current USD balance.
        """
        return self.balance

    def get_position(self):
        """
        Get the current open position.

        Returns
        -------
        dict or None
            The current position dictionary or None if no position is open.
        """
        return self.position
