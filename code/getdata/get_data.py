import yfinance as yf
import pandas as pd
import datetime
# Define your list of symbols with correct Yahoo Finance tickers

def get_yfinance_data(features=[
    "SPY",
    "XB1",
    "TLT",
    "GOLD",  # Gold Futures
    "VIXM"   # RBOB Gasoline Futures
],start_date="1994-01-01",end_date=datetime.date.today() ):

    data = yf.download(features, start=start_date, end=end_date,period="1d")
    data.to_csv("download.csv")
    
    
your_api_key = "d6c94d3495ef4ffea2cd239d08447db4"
import pandas as pd
import requests
import time
from datetime import datetime, timedelta

def get_tw_data(symbols=[
    "IVV",

    "VIXM",
    "TLT",
    "GOLD",
    "RB=F"
], start_date="2000-01-01", end_date=datetime.today(), your_api_key="YOUR_API_KEY", output_file="financial_data2.csv"):

    base_url = "https://api.twelvedata.com/time_series"
    all_symbol_data = {}

    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        symbol_data = pd.DataFrame()
        fetched_datetimes = set()
        fetch_end_date = None

        while True:
            params = {
                "symbol": symbol,
                "interval": "1min",
                "apikey": your_api_key,
                "outputsize": 5000,
                "format": "JSON",
            }

            if fetch_end_date:
                params["end_date"] = (pd.to_datetime(fetch_end_date) - timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')

            try:
                response = requests.get(base_url, params=params)
                time.sleep(8)
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as e:
                print(f"Network or API error for {symbol}: {e}")
                break
            except ValueError as e:
                print(f"JSON decoding error for {symbol}: {e}")
                break

            if "values" not in result or not result["values"]:
                print(f"No more data for {symbol}.")
                print(result)
                break

            df = pd.DataFrame(result["values"])
            print(len(df))
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df[~df['datetime'].isin(fetched_datetimes)].copy()

            if df.empty:
                print(f"No new data for {symbol}.")
                break

            fetched_datetimes.update(df['datetime'])
            symbol_data = pd.concat([symbol_data, df], ignore_index=True)
            fetch_end_date = df['datetime'].min()

        if symbol_data.empty:
            continue

        symbol_data = symbol_data.sort_values("datetime").set_index("datetime")
        symbol_data = symbol_data.astype({
            "open": "float",
            "high": "float",
            "low": "float",
            "close": "float",
            "volume": "float"
        })

        # Store in dict with MultiIndex preparation
        all_symbol_data[symbol] = symbol_data[["close", "open", "high", "low", "volume"]]

    # Combine all into one MultiIndex DataFrame
    combined = pd.concat(all_symbol_data, axis=1)
    combined.columns = pd.MultiIndex.from_tuples([(col, sym) for sym in all_symbol_data for col in all_symbol_data[sym].columns])

    # Sort columns for readability
    combined = combined.sort_index(axis=1, level=0)

    # Save to CSV
    combined.to_csv(output_file)
    print(f"Data saved to {output_file}")

    return combined

if __name__ == "__main__":
    get_tw_data(your_api_key=your_api_key)