import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np
from tqdm import tqdm
tqdm.pandas()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

events = pd.read_parquet("data/events.parquet")
spy = pd.read_parquet("data/spy.parquet").set_index('datetime')


device_num = 0 if device.type == 'mps' else -1
finbert_pipeline = pipeline(
    "sentiment-analysis",
    model="yiyanghkust/finbert-tone",
    tokenizer="yiyanghkust/finbert-tone",
    device=device_num
)

texts = events['text'].tolist()
batch_size = 32

print("Processing FinBERT sentiment analysis in batches...")
results = []
for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
    batch = texts[i:i + batch_size]
    batch_results = finbert_pipeline(batch)
    results.extend(batch_results)
    
def get_finbert_polarity(result):
    label = result['label'].toLower()
    score = result['score']
    if label == 'positive':
        return score
    elif label == 'negative':
        return -score
    else:
        return 0.0
    
events['finbert_polarity'] = [get_finbert_polarity(res) for res in results]


from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

print("Extracting SBERT embeddings...")
embeddings = sbert_model.encode(
    events['text'].tolist(),
    batch_size=32,
    show_progress_bar=True,
    device=device
)

np.save("data/events_embeddings.npy", embeddings)


topic_keywords = {
    'china': ['china', 'beijing'],
    'fed': ['fed', 'federal reserve'],
    'spy': ['spy', 's&p', 'market'],
}

def get_topic_flags(text):
    text_lower = text.lower()
    return {topic: int(any(word in text_lower for word in keywords)) 
            for topic, keywords in topic_keywords.items()}

topic_df = events['text'].apply(get_topic_flags).apply(pd.Series)
events = pd.concat([events, topic_df], axis=1)

def get_lag1_return(event_time):
    if pd.isna(event_time) or event_time not in spy.index:
        return np.nan
    # 30 minutes earlier
    prev_time = event_time - pd.Timedelta(minutes=30)
    pos = spy.index.searchsorted(prev_time, side='right') - 1
    if pos >= 0:
        prev_price = spy.iloc[pos]['close']
        now_price = spy.loc[event_time]['close']
        return (now_price - prev_price) / prev_price
    else:
        return np.nan
    
print("Calculating lag-1 SPY returns...")


events['lag1_spy_return'] = events['event_time'].apply(get_lag1_return)
events['large_move'] = (events['spy_30min_return'].abs() >= 0.003).astype(int)

print("Saving events with features...")

events.to_parquet('data/events_with_features.parquet')
print(f"Saved {len(events)} events with features to data/events_with_features.parquet")



