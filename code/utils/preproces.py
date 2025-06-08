import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

def preprocesDf(df:pd.DataFrame,target_col:str):


    ## change target close values for future log returns
    # Sample assumption: target_columns contains non-price columns like 'Date', 'Ticker', etc.

    df=df.dropna()
    target_col="close IVV"
    print(df.index)
    
        
    # Create future time-shifted DataFrames
    future_week = df[[target_col]].copy()
    future_week.index = future_week.index + pd.Timedelta(days=7)
    future_week = future_week.rename(columns={target_col: 'future_week_price'})

    future_month = df[[target_col]].copy()
    future_month.index = future_month.index + pd.Timedelta(days=30)
    future_month = future_month.rename(columns={target_col: 'future_month_price'})

    #future_year = df[[target_col]].copy()
    #future_year.index = future_year.index + pd.Timedelta(days=90)
    #future_year = future_year.rename(columns={target_col: 'future_year_price'})

    # Join back to original df
    #df = df.join(future_week, how='left')
    #df = df.join(future_month, how='left')
   # df = df.join(future_year, how='left')

    # Calculate log returns
    #df["target_next_week_return"] = np.log(df["future_week_price"] / df[target_col])
    #df["target_next_month_return"] = np.log(df["future_month_price"] / df[target_col])
    #df["target_next_year_return"] = np.log(df["future_year_price"] / df[target_col])
    
    # Drop intermediate price columns
   # df.drop(columns=["future_week_price",
    #                 "future_month_price", 
     #                #"future_year_price"
      #               ], inplace=True)
    
    
    for target in df.columns:
        
        
        if df[target].dtype != float:
            continue

        
        prices = df[target].values
        log_returns = np.full_like(prices, fill_value=np.nan, dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
        # Compute log returns using current price and previous day's price
            log_returns[1:] = np.log(np.where(prices[:-1] != 0, prices[1:] / prices[:-1], 1))
        if(target==target_col):
            df[f"target_{target}"] = log_returns
        df[target] = log_returns
    print(np.shape(df))
    
    df=computeFeatures(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df=df.fillna(0)
    
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df[(df.T != 0).any()]  
    print(np.shape(df))  
    return df

def computeFeatures(df):
    # Create features

    




    # Add time-based cyclical encodings

    
    df=compute_time_feat(df)

    #df = df.dropna()  # Remove rows with NaNs from rolling/lags

    return df



def trainTestSplit(df,label_columns):
    n = len(df)
    train_df = df[0 : int(n * 0.7)]
    val_df = df[int(n * 0.7) : int(n * 0.9)]
    test_df = df[int(n * 0.9) : ]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, test_df, val_df, train_mean[label_columns], train_std[label_columns]

def compute_time_feat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cyclical time features (sin/cos transformations) from a 'datetime' column
    and adds them to a new DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'datetime' column
                           in '%Y-%m-%d' format. The 'datetime' column will be
                           removed from the input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame containing the computed time features.
    """
    # Convert 'datetime' column to datetime objects and remove it from the original df
    date_time = df.index
    



    # 1. Day of Week (Monday=0, Sunday=6)
    # Cycle length = 7 days
    day_of_week = date_time.dayofweek
    df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    # 2. Day of Month (1-31)
    # Cycle length varies, use 31 as max for simplicity
    day_of_month = date_time.day
    df['day_of_month_sin'] = np.sin(2 * np.pi * day_of_month / 31)
    df['day_of_month_cos'] = np.cos(2 * np.pi * day_of_month / 31)

    # 3. Month (1-12)
    # Cycle length = 12 months
    month = date_time.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

    # 4. Day of Year (1-366 for leap years, 1-365 for non-leap years)
    # Cycle length = 365 or 366. Account for leap years for accuracy.
    day_of_year = date_time.dayofyear
    is_leap_year = date_time.is_leap_year
    days_in_year = np.where(is_leap_year, 366, 365)
    df['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / days_in_year)
    df['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / days_in_year)

    
    if False:
        hour = date_time.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        # Optional: Minute of Hour (0-59) - only if your datetime column includes time
        minute = date_time.minute
        df['minute_sin'] = np.sin(2 * np.pi * minute / 60)
        df['minute_cos'] = np.cos(2 * np.pi * minute / 60)

    return df

