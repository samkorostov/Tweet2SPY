# Predicting the S&P 500 off of Trump's Twitter

## **Project Goal**
This is a fun little side project to see if it's possible to use NLP and sentiment analysis to create a tweet-driven model that can flag >= 0.30% SPY moves in the next 30-minutes better than chance.

## **Road Map**
- [] Ensure model meets success criteria of **ROC-AUC >= 0.60** on 2023-24 hold-out set.
- [] Multi-class classification to predict sign of SPY moves
- [] Integration with Alpha Vantage API for real-time prediction

## **Data Sources**
- **Tweets:**
    - [Kaggle Trump Tweet Archive CSV] (https://www.kaggle.com/datasets/headsortails/trump-twitter-archive/data)
- **Market Prices:**
    - [1-minute SPY bars CSV] (https://www.kaggle.com/datasets/gratefuldata/intraday-stock-data-1-min-sp-500-200821)

## **Pipeline Overview**
1. **Environment Setup:**  
   - Python 3.10+, Conda or venv.
   - Install dependencies: `pandas`, `pyarrow`, `transformers`, `sentence-transformers`, `xgboost`, `scikit-learn`, `tqdm`, etc.

2. **Data Ingestion and Alignment:**  
   - Load SPY 1-min bars and Trump tweets.
   - Snap each tweet to the next market minute (`event_time`).
   - Calculate realized 30-min SPY return after each tweet.
   - Label events with `large_move` (abs(return) ≥ 0.3%).

3. **Feature Engineering:**  
   - **FinBERT polarity:** Financial sentiment score for each tweet.
   - **SBERT embedding:** 768-d vector for each tweet’s meaning.
   - **Topic flags:** Presence/absence of keywords (e.g., “china”, “fed”, “spy”).
   - **Lagged SPY return:** SPY return in previous 30 minutes.

4. **Model Training & Evaluation:**  
   - Concatenate all features.
   - **Binary classification**: Predict if a tweet is followed by a large SPY move.
   - Use **time-based split** (train on 2014–2022, test on 2023–24).
   - Evaluate with ROC-AUC, accuracy, and confusion matrix.
5. **Success Criteria:**
    - **ROC-AUC >= 0.60** on 2023-24 hold-out set.


