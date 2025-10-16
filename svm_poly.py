import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y= mnist["data"], mnist["target"]

some_digit = X[0]

y = y.astype(np.uint8) # convert to int

# Split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', svm.SVC(random_state=42))
])
param_grid = {
    'kernel': ['poly'],
    'C': [1, 10, 100, 1000],
    'degree': [2, 3, 4, 5],
    'coef0': [0.1, 0.5, 1.0]
}


grid_search = GridSearchCV(svm_clf, param_grid, cv=5,
                           scoring="accuracy",
                           verbose=2)


# Timing training
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()


print(f"Grid Search Runtime: {end_time - start_time:.2f} seconds")
print("Best parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

# Evaluate on test set
best_svm = grid_search.best_estimator_

start_pred_time = time.time()
y_pred = best_svm.predict(X_test)
end_pred_time = time.time()

test_acc = accuracy_score(y_test, y_pred)
print(f"Prediction Runtime: {end_pred_time - start_pred_time:.2f} seconds")
print("Test Accuracy:", test_acc)

