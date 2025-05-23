import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def simple_backtest(results_df: pd.DataFrame, real_column: str, forecast_column: str):

    
    model_results_df = results_df.copy()

    # Strategy: take the return only if forecast is positive
    model_results_df['strategy_return'] = np.where(
        model_results_df[forecast_column] > 0,
        model_results_df[real_column],
        0.0
    )

    # Compute cumulative returns using compounding
    model_results_df['strategy_cum_return'] = (model_results_df['strategy_return']).cumsum()
    model_results_df['buy_hold_cum_return'] = ( model_results_df[real_column]).cumsum()

    # Final cumulative return
    model_cumulative_return = model_results_df['strategy_cum_return'].iloc[-1] - 1
    model_stddev = model_results_df['strategy_return'].std()

    buy_hold_cumulative_return = model_results_df['buy_hold_cum_return'].iloc[-1] - 1
    buy_hold_stddev = model_results_df[real_column].std()

    # Plot cumulative returns
    plt.figure(figsize=(10, 6))
    plt.plot(model_results_df['strategy_cum_return'], label='Strategy')
    plt.plot(model_results_df['buy_hold_cum_return'], label='Buy & Hold', linestyle='--')
    plt.title('Cumulative Returns')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return (Total %)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Count number of buy operations (forecast > 0)
    buy_signals = model_results_df[forecast_column] > 0
    num_buy_ops = buy_signals.sum()
    num_sell_ops = len(model_results_df) - num_buy_ops

    # Calculate winning trades (positive return when forecast > 0)
    winning_trades = (model_results_df['strategy_return'] > 0) & buy_signals
    percent_winning = (winning_trades.sum() / num_buy_ops * 100) if num_buy_ops > 0 else 0

    print(f"Number of buy operations: {num_buy_ops}, sell operations: {num_sell_ops}")
    print(f"Percentage of winning operations: {percent_winning:.2f}%")
    print(f"Model cumulative return: {model_cumulative_return:.4f}, Buy & Hold: {buy_hold_cumulative_return:.4f}")
    print(f"Model std dev: {model_stddev:.4f}, Buy & Hold std dev: {buy_hold_stddev:.4f}")
