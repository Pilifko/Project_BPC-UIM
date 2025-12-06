# -*- coding: utf-8 -*-
# importování modulů:
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier, Pool
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data(csv_data: str) -> pd.DataFrame:
    """
    Načte data z CSV souboru.
    """
    data = pd.read_csv(csv_data)
    return data


def data_preprocessing(data: pd.DataFrame) -> tuple:
    """
    Vyčistí data, odstraní odlehlé hodnoty a nepotřebné sloupce (chol, fbs).
    Vrací X (features) a y (target).
    """
    df = data.copy()

    target = df['target']

    del df['target']
    del df['fbs']
    del df['chol']

    df['age'] = df['age'].apply(lambda x: int(x) if 0 <= x <= 120 else np.nan).astype(float)
    df['sex'] = df['sex'].apply(lambda x: x if x in [0, 1] else np.nan).astype(float)
    df['cp'] = df['cp'].apply(lambda x: x if x in [0, 1, 2, 3] else np.nan).astype(float)
    df['trestbps'] = df['trestbps'].apply(lambda x: int(x) if 100 <= x <= 300 else np.nan).astype(float)
    df['restecg'] = df['restecg'].apply(lambda x: x if x in [0, 1, 2] else np.nan).astype(float)
    df['thalach'] = df['thalach'].apply(lambda x: int(x) if 50 <= x <= 250 else np.nan).astype(float)
    df['exang'] = df['exang'].apply(lambda x: x if x in [0, 1] else np.nan).astype(float)
    df['oldpeak'] = df['oldpeak'].apply(lambda x: int(x) if 0 <= x <= 3 else np.nan).astype(float)
    df['slope'] = df['slope'].apply(lambda x: x if x in [0, 1, 2] else np.nan).astype(float)
    df['ca'] = df['ca'].apply(lambda x: x if x in [0, 1, 2, 3] else np.nan).astype(float)
    df['thal'] = df['thal'].apply(lambda x: x if x in [0, 1, 2, 3] else np.nan).astype(float)

    return df, target

def impute_data(X,X_test: Optional[pd.DataFrame] = None) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imputuje chybějící hodnoty.
    """
    cols = X.columns

    imputer = IterativeImputer(random_state=42, max_iter=10)
    X_imputed = imputer.fit_transform(X)

    X_final = pd.DataFrame(X_imputed, columns=cols)
    X_final = process_data(X_final)

    if X_test is not None:
        cols_test = X_test.columns
        X_test_imputed = imputer.transform(X_test)
        X_final_test = pd.DataFrame(X_test_imputed, columns=cols_test)
        X_final_test = process_data(X_final_test)
        return X_final, X_final_test
    else:
        return X_final


def process_data(X) -> DataFrame:
    """
    Zpracuje imputovaná data.
    """

    cat_cols = ['sex', 'cp', 'restecg', 'thal']
    cols_cat = [c for c in cat_cols if c in X.columns]

    X[cols_cat] = X[cols_cat].round()
    X['oldpeak'] = X['oldpeak'].round(1)
    X['sex'] = X['sex'].astype(int)
    X['cp'] = X['cp'].astype(int)
    X['restecg'] = X['restecg'].astype(int)
    X['thal'] = X['thal'].astype(int)

    return X

def train_final_model(X_train: pd.DataFrame, y_train: pd.Series) -> CatBoostClassifier:
    """
    Natrénuje finální model s nejlepšími parametry a vykresluje loss fci.
    """
    #full_params = params.copy()
    full_params={
        'iterations': 245,
        'depth': 2,
        'learning_rate': 0.023,
        'l2_leaf_reg': 11.6,
        'loss_function': 'Logloss',
        'verbose': 100,
        'random_seed': 42,
        'allow_writing_files': False,
        'cat_features': ['sex', 'cp', 'restecg', 'thal'],
        'one_hot_max_size': 10
    }

    model = CatBoostClassifier(**full_params)

    model.fit(X_train, y_train)
    return model


def compute_statistics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Vypočítá metriky modelu.
    """
    stats = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average='weighted'),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Confusion Matrix": confusion_matrix(y_true, y_pred)
    }
    return stats


def main(csv_path: str) -> None:
    """
    Hlavní funkce, která řídí tok programu.
    """
    # 1. Načtení dat
    try:
        raw_data = load_data(csv_path)
    except FileNotFoundError:
        print(f"Chyba: Soubor '{csv_path}' nebyl nalezen.")
        return

    # 2. Preprocessing (čištění)
    X, y = data_preprocessing(raw_data)
    print(f"Data po vyčištění: {X.shape}, Features: {list(X.columns)}")

    # 3. Rozdělení na Train a Test (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=22, stratify=y
    )
    # 4. Process (Imputace)
    X_train_processed, X_test_processed = impute_data(X_train, X_test)

    # 5. Trénink finálního modelu
    print("\n--- Trénink finálního modelu ---")
    model = train_final_model(X_train_processed, y_train)

    # 6. Evaluace
    y_pred = model.predict(X_test_processed)

    stats = compute_statistics(y_test, y_pred)

    print("\n--- Výsledky modelu (Test set) ---")
    print(f"Matthews Correlation Coefficient (MCC): {stats['MCC']:.4f}")
    print(f"Accuracy: {stats['Accuracy']:.4f}")
    print(f"F1 Score: {stats['F1 Score']:.4f}")
    print("\nConfusion Matrix:")
    print(stats['Confusion Matrix'])


    y_pred_full = model.predict(impute_data(X))
    stats_full = compute_statistics(y, y_pred_full)

    print("\n--- Výsledky modelu (Full dataset) ---")
    print(f"Matthews Correlation Coefficient (MCC): {stats_full['MCC']:.4f}")
    print(f"Accuracy: {stats_full['Accuracy']:.4f}")
    print(f"F1 Score: {stats_full['F1 Score']:.4f}")

    model.save_model('heart_disease_prediction_model')

if __name__ == "__main__":
    main('heart-disease_data.csv')