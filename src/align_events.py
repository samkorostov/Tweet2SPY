import pandas as pd

def preprocess_tweets(path_to_csv, out_parquet_path):
    tweets = pd.read_csv(path_to_csv)
    tweets = tweets[tweets['is_retweet'] == False]
    tweets = tweets.rename(columns={'datetime': 'created_at'})
    tweets = tweets[['id', 'text', 'created_at']]
    tweets['created_at'] = pd.to_datetime(tweets['created_at'], utc=True, errors='coerce')
    tweets = tweets.dropna(subset=['created_at', 'text'])
    tweets = tweets.sort_values('created_at').reset_index(drop=True)
    tweets.to_parquet(out_parquet_path)
    print(f"Saved {len(tweets)} cleaned tweets to {out_parquet_path}")
    
def preprocess_spy(path_to_csv, out_parquet_path):
    spy = pd.read_csv(path_to_csv)
    spy['date'] = pd.to_datetime(spy['date'], utc=True, errors='coerce')
    spy = spy.rename(columns={'date': 'datetime'})
    spy = spy[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    spy = spy.dropna(subset=['datetime', 'close'])
    spy = spy.sort_values('datetime').reset_index(drop=True)
    spy.to_parquet(out_parquet_path)
    print(f"Saved {len(spy)} cleaned SPY data to {out_parquet_path}")
    

tweets = pd.read_parquet("data/tweets.parquet")
spy = pd.read_parquet("data/spy.parquet")
spy = spy.sort_values('datetime').set_index('datetime')
spy = spy[~spy.index.duplicated(keep='first')]  # Remove duplicates

spy_idx = spy.index

def snap_to_next_minute(ts):
    pos = spy_idx.searchsorted(ts, side='left')
    if pos < len(spy_idx):
        return spy_idx[pos]
    else:
        return pd.NaT

tweets['event_time'] = tweets['created_at'].apply(snap_to_next_minute)

import numpy as np
def calc_30min_return(event_time):
    if pd.isna(event_time) or event_time not in spy.index:
        return np.nan
    start_price = spy.loc[event_time, 'close']
    end_time = event_time + pd.Timedelta(minutes=30)
    pos = spy_idx.searchsorted(end_time, side='left')
    if pos < len(spy_idx):
        end_price = spy.iloc[pos]['close']
        return (end_price - start_price) / start_price
    else:
        return np.nan
    
tweets['spy_30min_return'] = tweets['event_time'].apply(calc_30min_return)
events = tweets.dropna(subset=['event_time', 'spy_30min_return']).copy()


print(events['spy_30min_return'].apply(type).value_counts())
print(events['spy_30min_return'].head(10))


events.to_parquet("data/events.parquet")
print(f"Saved {len(events)} events with SPY returns to data/events.parquet")


events = pd.read_parquet('data/events.parquet')
spy = pd.read_parquet('data/spy.parquet').set_index('datetime')

# Print 5 random aligned events
sample = events.sample(5, random_state=42)
for _, row in sample.iterrows():
    tweet_time = row['created_at']
    snap_time = row['event_time']
    print(f"TWEET: {tweet_time}  -->  EVENT_TIME: {snap_time}")
    print(f"Tweet text: {row['text']}")
    # Check SPY bar for snap_time
    if pd.notna(snap_time) and snap_time in spy.index:
        bar = spy.loc[snap_time]
        print(f"SPY at {snap_time}: open={bar['open']}, close={bar['close']}")
    print("-" * 60)
    