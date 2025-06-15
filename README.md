# Predicting the S&P 500 off of Trump's Twitter

## **Project Goal**
This is a fun little side project to see if it's possible to use NLP and sentiment analysis to create a tweet-driven model that can flag >= 0.30% SPY moves in the next 30-minutes better than chance.

## **Results**

- **Test period:** 2021 (fully out-of-sample, last portion of dataset)
- **Test set size:** 6,402 events (1,235 “large move” cases)
- **Model performance:**
    - **ROC-AUC:** `0.59`
    - **Accuracy:** `0.79`
    - **Confusion Matrix:**
        ```
        [[4899  268]
         [1080  155]]
        ```
    - **Classification Report (key metrics for "large move" class):**
        - **Precision:** `0.37`
        - **Recall:** `0.13`
        - **F1-score:** `0.19`

**Interpretation:**  
The model demonstrates weak but genuine predictive signal (ROC-AUC ~0.59) for large (≥ 0.3%) 30-minute SPY moves triggered by tweets. About 1 in 3 flagged “large move” events was correct (precision), and the model caught 1 in 8 of all true large moves (recall). Results are consistent with financial ML baselines for rare event prediction and highlight the difficulty of forecasting market jumps from news or social media, even with modern NLP.

**Limitations:**
- Only predicts *if* the SPY will change, not the direction it will change
- Data is sourced soley off of Trump's Twitter,  including other prominent user's accounts could make it more accurate

## **Summary**
So does this work? **Kinda?** It is better than random guessing (although not by much), but to be fully honest I didn't expect any of it to work so that's a win. If you are expecting to make money off of this, that's not gonna happen-but as a little side project I'd say this works great.


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
   - Use **time-based split** (train on 2009–2019, test on 2020-2021).
   - Evaluate with ROC-AUC, accuracy, and confusion matrix.
5. **Success Criteria:**
    - **ROC-AUC >= 0.60** on 2023-24 hold-out set.


