from catboost import CatBoostClassifier
from pandas.io.xml import preprocess_data
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
from main import *
from main import load_data

X, y = data_preprocessing(load_data("heart-disease_data.csv"))


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

model = CatBoostClassifier(
    iterations=400,
    learning_rate=0.03,
    depth=3,
    loss_function='Logloss',
    eval_metric='F1',
    verbose=False,
    random_state=39,
)

y_pred = cross_val_predict(model, X, y, cv=kfold)

print("Accuracy :", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred))
print("Recall   :", recall_score(y, y_pred))
print("F1-score :", f1_score(y, y_pred))
print("Confusion matrix:\n", confusion_matrix(y, y_pred))

model.fit(X, y)
