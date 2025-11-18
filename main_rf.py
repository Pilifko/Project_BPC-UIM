import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------------------------------------------------------------
# 1) Načtení dat
# ---------------------------------------------------------------------
# X = dataframe nebo numpy pole tvaru (250, 13)
# y = vektor 0/1 pro zdravý/nemocný

# Příklad:
df = pd.read_csv("heart-disease_data.csv")
X = df.iloc[:, :-1].values   # prvních 13 sloupců
y = df.iloc[:, -1].values    # cílová proměnná

# ---------------------------------------------------------------------
# 2) K-fold cross-validace — doporučeno u malých datasetů
# ---------------------------------------------------------------------
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Cross-validated predikce
y_pred = cross_val_predict(model, X, y, cv=kfold)

# ---------------------------------------------------------------------
# 3) Výpočet metrik
# ---------------------------------------------------------------------
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)
print("Confusion matrix:")
print(cm)

# ---------------------------------------------------------------------
# 4) Trénování finálního modelu (na celém datasetu)
# ---------------------------------------------------------------------
model.fit(X, y)
print("\nModel je natrénován a připraven k použití!")
