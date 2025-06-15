from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import time
import os


API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
SYMBOL = "SPY"
INTERVAL = "1min"
OUTPUT_FILE = "data/spy.parquet"

def fetch_latest():
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    df, meta = ts.get_intraday(symbol=SYMBOL, interval=INTERVAL, outputsize='full')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

