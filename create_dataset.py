import requests
import pandas as pd
import pandas_ta as ta
import os  # or from pathlib import Path


your_api_key = "d6c94d3495ef4ffea2cd239d08447db4"
DATASETFILE="dataset.csv"
# symbols: list of strings
# features: dict with symbol as key and list of feature names as values

def createDataset(features, target_features):
    if os.path.exists(DATASETFILE):  
        df = pd.read_csv(DATASETFILE)
        return df
    else:
    
        pass
    merged_df = None
    data_cache = {}

    def fetch_and_prepare(symbol):
        print(f"\nFetching data for {symbol}...")
        url = f"https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": "1day",
            "apikey": your_api_key,
            "outputsize": 5000,
            "format": "JSON"
        }

        response = requests.get(url, params=params)
        result = response.json()

        if "values" not in result:
            print(f"Failed to fetch data for {symbol}: {result.get('message', 'Unknown error')}")
            return None

        df = pd.DataFrame(result["values"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values("datetime")
        df = df.astype({
            "open": "float",
            "high": "float",
            "low": "float",
            "close": "float",
            "volume": "float"
        })

        return df

    # Process feature symbols
    for symbol in features.keys():
        df = fetch_and_prepare(symbol)
        if df is None:
            continue

        feature_list = features.get(symbol, [])

        # Compute indicators
        if "RSI" in feature_list:
            df["RSI"] = ta.rsi(df["close"], length=14)
        if "OBV" in feature_list:
            df["OBV"] = ta.obv(df["close"], df["volume"])
        if "MACD" in feature_list:
            macd = ta.macd(df["close"])
            df["MACD"] = macd["MACD_12_26_9"]
            df["MACD_signal"] = macd["MACDs_12_26_9"]
            df["MACD_hist"] = macd["MACDh_12_26_9"]
        if "MA50" in feature_list:
            df["MA50"] = ta.sma(df["close"], length=50)
        if "MA100" in feature_list:
            df["MA100"] = ta.sma(df["close"], length=100)
        if "MA200" in feature_list:
            df["MA200"] = ta.sma(df["close"], length=200)
        if "Bollinger" in feature_list:
            bb = ta.bbands(df["close"], length=20)
            df["BBL"] = bb["BBL_20_2.0"]
            df["BBM"] = bb["BBM_20_2.0"]
            df["BBU"] = bb["BBU_20_2.0"]

        df.columns = df.columns.map(lambda col: col if col == "datetime" else f"{col}_{symbol}")
        data_cache[symbol] = df

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="datetime", how="outer")

    # Process target-only symbols
    for symbol in target_features.keys():
        if symbol not in data_cache:
            df = fetch_and_prepare(symbol)
            if df is None:
                continue
            data_cache[symbol] = df

        df = data_cache[symbol]

        # Keep only datetime + specified target columns
        target_cols = target_features[symbol]
        target_cols = [col if col == "datetime" else f"{col}_{symbol}" for col in target_features[symbol]]

        keep_cols = ["datetime"] + target_cols
        df = df[keep_cols]

        # Rename with symbol and prefix "TARGET_"
        df.columns = [
            col if col == "datetime" else f"TARGET_{col}"
            for col in df.columns
        ]

        merged_df = pd.merge(merged_df, df, on="datetime", how="outer")
    merged_df.to_csv("dataset.csv",index=False)
    return merged_df
# Example usage:

features = {
    "AAPL": ["MA100", "MA200", "volume", "RSI", "Bollinger"],
    "GOOGL": ["MACD", "OBV", "RSI"]
}

targets_features = {
    "AAPL": ["close"],
   
}
dataset = createDataset(features,targets_features)

# Display sample
print(dataset.tail(10))
