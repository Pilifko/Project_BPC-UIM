from cb import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from main import *

X, y = data_preprocessing(load_data("heart-disease_data.csv"))


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

model = CatBoostClassifier(random_strength=2, learning_rate=0.01, l2_leaf_reg=7, iterations=1000, depth=4, border_count=32, verbose=False)

y_pred = cross_val_predict(model, X, y, cv=kfold)

print("Accuracy :", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred))
print("Recall   :", recall_score(y, y_pred))
print("F1-score :", f1_score(y, y_pred))
print("Confusion matrix:\n", confusion_matrix(y, y_pred))

model.fit(X, y)
