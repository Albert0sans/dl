import pandas as pd
from strategy import Strategy

class BackTest:
    def __init__(self, strategy:Strategy, data: pd.DataFrame, initial_cash: float = 100000.0):
        """
        Initialize the BackTest instance.
        
        Args:
            strategy (Strategy): An instance of the Strategy class.
            data (pd.DataFrame): Historical market data indexed by datetime.
            initial_cash (float): Initial capital for the backtest.
        """
        self.strategy = strategy
        self.data = data
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # symbol -> quantity
        self.portfolio_history = []  # Track portfolio value over time
        self.current_date = None

    def run(self):
        """
        Run the backtest loop over all data points.
        """
        for current_date, row in self.data.iterrows():
            self.current_date = current_date
            signals = self.strategy.generate_signals(row)
            self.execute_signals(signals, row)
            self.update_portfolio_value(row)

    def execute_signals(self, signals: dict, row: pd.Series):
        """
        Execute trade signals.

        Args:
            signals (dict): e.g., {'AAPL': 'buy', 'GOOG': 'sell'}.
            row (pd.Series): Current market data row.
        """
        for symbol, action in signals.items():
            price = row.get(symbol)
            if action == 'buy':
                self.buy(symbol, price)
            elif action == 'sell':
                self.sell(symbol, price)

    def buy(self, symbol: str, price: float):
        """
        Execute a buy order.

        Args:
            symbol (str): Asset ticker.
            price (float): Execution price.
        """
        if price and self.cash >= price:
            quantity = int(self.cash // price)
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            self.cash -= quantity * price

    def sell(self, symbol: str, price: float):
        """
        Execute a sell order.

        Args:
            symbol (str): Asset ticker.
            price (float): Execution price.
        """
        quantity = self.positions.get(symbol, 0)
        if price and quantity > 0:
            self.cash += quantity * price
            self.positions[symbol] = 0

    def update_portfolio_value(self, row: pd.Series):
        """
        Update portfolio value and record it.

        Args:
            row (pd.Series): Current market data row.
        """
        value = self.cash
        for symbol, quantity in self.positions.items():
            price = row.get(symbol)
            if price:
                value += quantity * price
        self.portfolio_history.append((self.current_date, value))

    def get_performance_metrics(self) -> dict:
        """
        Compute performance metrics.

        Returns:
            dict: Portfolio metrics.
        """
        df = pd.DataFrame(self.portfolio_history, columns=["Date", "Value"]).set_index("Date")
        returns = df["Value"].pct_change().dropna()
        total_return = df["Value"].iloc[-1] / self.initial_cash - 1
        sharpe_ratio = returns.mean() / returns.std() * (252**0.5) if not return*_
