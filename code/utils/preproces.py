import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

def preprocesDf(df):


    ## change target close values for future log returns
    # Sample assumption: target_columns contains non-price columns like 'Date', 'Ticker', etc.

    
    for target in df.columns:
        
        print(df[target].dtype)
        if df[target].dtype != float:
            continue

        
        prices = df[target].values
        log_returns = np.full_like(prices, fill_value=np.nan, dtype=np.float64)
        print(prices)
        # Compute log returns using current price and previous day's price
        log_returns[1:] = np.log(prices[1:] /prices[:-1])

        # Optionally: create a new column to store the log returns
        df[f"target_{target}"] = log_returns
    
    df=df.dropna()

    
    return df

def computeFeatures(df):
    # Create features
    print(df.columns)
    date_time = pd.to_datetime(df.pop("datetime"), format="%Y-%m-%d")

    lagged_features = []

    for col in df.columns:
        lag_dict = {
            f"{col}_mean_7": df[col].shift(1).rolling(7).mean(),
            f"{col}_mean_30": df[col].shift(1).rolling(30).mean(),
            f"{col}_mean_200": df[col].shift(1).rolling(200).mean(),
            f"{col}_std_7": df[col].rolling(7).std(),
            f"{col}_std_30": df[col].rolling(30).std(),
            f"{col}_std_200": df[col].rolling(200).std()
        }

        # Compute log returns for volatility
        log_returns = np.log(df[col] / df[col].shift(1))

        lag_dict.update({
            "support_1": df[col].rolling(window=20).min(),
            "support_2": df[col].rolling(window=50).min(),
            "support_3": df[col].rolling(window=100).min(),

            "resistance_1": df[col].rolling(window=20).max(),
            "resistance_2": df[col].rolling(window=50).max(),
            "resistance_3": df[col].rolling(window=100).max(),

            f"{col}_rolling_volatility_14": log_returns.rolling(14).std(),
            f"{col}_rolling_volatility_30": log_returns.rolling(30).std(),
        })

        # Volume-specific features placeholder
        if col.lower().startswith('volume'):
            lag_dict.update({
                # Add volume-related stats here if needed
            })

        lagged_features.append(pd.DataFrame(lag_dict))

    # Add time-based cyclical encodings
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day
    month = year / 12

    df["month_sin"] = np.sin(timestamp_s * (2 * np.pi / month))
    df["month_cos"] = np.cos(timestamp_s * (2 * np.pi / month))
    df["year_sin"] = np.sin(timestamp_s * (2 * np.pi / year))
    df["year_cos"] = np.cos(timestamp_s * (2 * np.pi / year))

    # Combine lag features
   # df_lags = pd.concat(lagged_features, axis=1)
   # df = pd.concat([df, df_lags], axis=1)

    df = df.dropna()  # Remove rows with NaNs from rolling/lags

    return df



def trainTestSplit(df):
    n = len(df)
    train_df = df[0 : int(n * 0.7)]
    val_df = df[int(n * 0.7) : int(n * 0.9)]
    test_df = df[int(n * 0.9) : ]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, test_df, val_df,train_mean,train_std

