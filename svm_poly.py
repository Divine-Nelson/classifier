import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

# --- Load MNIST dataset ---
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(np.uint8)

# --- Split ---
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# --- Create Pipeline ---
svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', svm.SVC(random_state=42))
])

# --- Define Parameter Grid ---
param_grid = {
    'svm__kernel': ['poly'],
    'svm__C': [1, 10, 100, 1000],
    'svm__degree': [2, 3, 4, 5],
    'svm__coef0': [0.1, 0.5, 1.0]
}

# --- Grid Search ---
grid_search = GridSearchCV(
    svm_clf,
    param_grid,
    cv=5,
    scoring="accuracy",
    verbose=2,
    n_jobs=-1
)

# --- Timing Training ---
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

print(f"\nGrid Search Runtime: {end_time - start_time:.2f} seconds")
print("Best parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

# --- Evaluate on Test Set ---
best_svm = grid_search.best_estimator_

start_pred_time = time.time()
y_pred = best_svm.predict(X_test)
end_pred_time = time.time()

test_acc = accuracy_score(y_test, y_pred)
print(f"\nPrediction Runtime: {end_pred_time - start_pred_time:.2f} seconds")
print("Test Accuracy:", test_acc)

# --- Extra Metrics ---
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Confusion Matrix ---
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), cmap='Blues', annot=False)
plt.title('Confusion Matrix - SVM (Polynomial Kernel)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

"""
Grid Search Runtime: 30959.44 seconds
Best parameters: {'svm__C': 1000, 'svm__coef0': 0.1, 'svm__degree': 3, 'svm__kernel': 'poly'}
Best CV Accuracy: 0.9793166666666668

Prediction Runtime: 53.73 seconds
Test Accuracy: 0.9809

Classification Report:
        precision    recall  f1-score   support

           0       0.99      0.99      0.99       980
           1       0.99      0.99      0.99      1135
           2       0.98      0.98      0.98      1032
           3       0.98      0.98      0.98      1010
           4       0.98      0.98      0.98       982
           5       0.98      0.98      0.98       892
           6       0.99      0.98      0.98       958
           7       0.98      0.97      0.98      1028
           8       0.97      0.98      0.97       974
           9       0.98      0.97      0.97      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000
"""