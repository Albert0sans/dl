import pandas as pd

class Strategy:
    def __init__(self):
        """
        Base Strategy class.
        Initialize any required parameters or state here.
        """
        pass

    def generate_signals(self, row: pd.Series) -> dict:
        """
        Generate trading signals for a single row of market data.

        Args:
            row (pd.Series): A row from the data DataFrame containing market data.

        Returns:
            dict: Signals in the form { 'TICKER': 'buy' / 'sell' / 'hold' }
        """
        raise NotImplementedError("generate_signals() must be implemented by subclass.")

    def reset(self):
        """
        Optional: Reset any internal state when restarting a backtest.
        """
        pass
