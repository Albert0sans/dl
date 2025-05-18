import pandas as pd
import numpy as np


def preprocesDf(df):
    
    target_columns = df.columns[df.columns.map(lambda col: col.startswith("TARGET"))]

    date_time = pd.to_datetime(df.pop("datetime"), format="%Y-%m-%d")
    
    

    
    #create features
    target_columns = list(target_columns)  # convert if it's an Index
    non_target_cols = [
        col for col in df.columns if col not in (target_columns)
    ]
    lagged_features = []

    for col in non_target_cols:
         lag_dict = {
       #      f"{col}_lag1": df[col].shift(1),
       #      f"{col}_lag7": df[col].shift(7),
       #      f"{col}_lag30": df[col].shift(30),
       #      f"{col}_mean_7": df[col].shift(1).rolling(7).mean(),
       #      f"{col}_mean_30": df[col].shift(1).rolling(30).mean(),
       #      f"{col}_std_7": df[col].rolling(7).std(),
             f"{col}_std_30": df[col].rolling(30).std()
             
         }
             ## compute 1 2 and 3 supprort prices:
             
                   
         if col.lower().startswith('close'):
              returns = df[col].pct_change()
              lag_dict.update({
         #           "support_1": df[col].rolling(window=20).min(),
         #           "support_2": df[col].rolling(window=50).min(),
         #           "support_3": df[col].rolling(window=100).min(),

        #            "resistance_1": df[col].rolling(window=20).max(),
        #            "resistance_2": df[col].rolling(window=50).max(),
        #            "resistance_3": df[col].rolling(window=100).max(),
                
        #          "rolling_volatility_14": returns.rolling(14).std(),
        #            "rolling_volatility_30": returns.rolling(30).std(),
                })

             
         if col.lower().startswith('volume'):
                lag_dict.update({
                   # "volume_max_all_time": df[col].expanding().max(),
                 #   "volume_max_365d": df[col].rolling(window=365).max(),
                #    "volume_max_30d": df[col].rolling(window=30).max(),
                })
             
             
         
         lagged_features.append(pd.DataFrame(lag_dict))
    
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = (365.2425) * day
    month = year/12

    df["day_sin"] = np.sin(timestamp_s * (2 * np.pi / month))
    df["day_cos"] = np.cos(timestamp_s * (2 * np.pi / month))
    
    
    df["year_sin"] = np.sin(timestamp_s * (2 * np.pi / year))
    df["year_cos"] = np.cos(timestamp_s * (2 * np.pi / year))
    
    # Concatenate all lag features at once
    df_lags = pd.concat(lagged_features, axis=1)
    df = pd.concat([df, df_lags], axis=1)
    

    df = df.dropna()  # Drop rows with NaNs from lagging
    #df.to_csv("test.csv")

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

    return train_df, test_df, val_df
