import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
import joblib

events = pd.read_parquet('data/events_with_features.parquet')
embeddings = np.load('data/events_embeddings.npy')

topic_flag_cols = ['china', 'fed', 'spy']
other_feature_cols = ['finbert_polarity', 'lag1_spy_return'] + topic_flag_cols

X_other = events[other_feature_cols].values
X = np.hstack([embeddings, X_other])
y = events['large_move'].values


events['event_time'] = pd.to_datetime(events['event_time'])
train_idx = events['event_time'] < '2023-10-01'
test_idx = events['event_time'] >= '2023-10-01'

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

n_negative = np.sum(y_train == 0)
n_positive = np.sum(y_train == 1)
scale_pos_weight = n_negative / n_positive

clf = XGBClassifier(
    max_depth=4,
    n_estimators=300,
    objective= 'binary:logistic',
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

clf.fit(X_train, y_train)

threshold = 0.390
y_pred_prob = clf.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= threshold).astype(int)

roc_auc = roc_auc_score(y_test, y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f'ROC AUC: {roc_auc:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(cm)
print('Classification Report:')
print(classification_report(y_test, y_pred))

joblib.dump(clf, 'data/xgb_large_move_model.joblib')
print("Model saved to data/xgb_large_move_model.joblib")



from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# y_test: true labels
# y_pred_prob: predicted probabilities (not thresholded)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


j_scores = tpr - fpr
best_index = j_scores.argmax()
optimal_threshold = thresholds[best_index]

print(f"Optimal threshold by Youden's J: {optimal_threshold:.3f}")