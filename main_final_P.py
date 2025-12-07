# -*- coding: utf-8 -*-
# Import necessary modules
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from typing import Optional

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data(csv_data: str) -> pd.DataFrame:
    """
    Loads data from csv file.

    Input:
    csv_data: csv file path

    Output:
    DataFrame from csv file
    """
    data = pd.read_csv(csv_data)
    return data


def data_preprocessing(data: pd.DataFrame) -> tuple:
    """
    Function to preprocess data.
    Deletes cols with very low correlation.

    Input:
    data: DataFrame to be preprocessed

    Output:
    tuple: data and their target values
    """
    target = data['target']

    del data['target']
    del data['fbs']
    del data['chol']

    data['age'] = data['age'].apply(lambda x: int(x) if 0 <= x <= 120 else np.nan).astype(float)
    data['sex'] = data['sex'].apply(lambda x: x if x in [0, 1] else np.nan).astype(float)
    data['cp'] = data['cp'].apply(lambda x: x if x in [0, 1, 2, 3] else np.nan).astype(float)
    data['trestbps'] = data['trestbps'].apply(lambda x: int(x) if 100 <= x <= 300 else np.nan).astype(float)
    data['restecg'] = data['restecg'].apply(lambda x: x if x in [0, 1, 2] else np.nan).astype(float)
    data['thalach'] = data['thalach'].apply(lambda x: int(x) if 50 <= x <= 250 else np.nan).astype(float)
    data['exang'] = data['exang'].apply(lambda x: x if x in [0, 1] else np.nan).astype(float)
    data['oldpeak'] = data['oldpeak'].apply(lambda x: int(x) if 0 <= x <= 3 else np.nan).astype(float)
    data['slope'] = data['slope'].apply(lambda x: x if x in [0, 1, 2] else np.nan).astype(float)
    data['ca'] = data['ca'].apply(lambda x: x if x in [0, 1, 2, 3] else np.nan).astype(float)
    data['thal'] = data['thal'].apply(lambda x: x if x in [0, 1, 2, 3] else np.nan).astype(float)

    return data, target


def impute_data(X: pd.DataFrame,
                X_test: Optional[pd.DataFrame] = None,
                ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imputes NaN values of Dataframe X and optionally X_test.

    Input:
    X: Dataframe to be imputed
    X_test: (optional) Dataframe to be imputed

    Output:
    X_final: imputed DataFrame X
    X_final_test: imputed DataFrame X_test, only if X_test is not None
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


def process_data(X:pd.DataFrame) -> pd.DataFrame:
    """
    Processes Dataframe X to contain correct value types.

    Input:
    X: Dataframe to be processed

    Output:
    X: processed Dataframe
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


def train_final_model(X_train: pd.DataFrame,
                      y_train: pd.Series) -> CatBoostClassifier:
    """
    Trains CatBoostClassifier model with training data.

    Input:
    X_train: training data
    y_train: training labels

    Output:
    model: trained CatBoostClassifier model
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
    Computes statistics of prediction accuracy.

    Input:
    y_true: true labels
    y_pred: predicted labels

    Output:
    stats: statistics of prediction accuracy
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
    Main function which contains the whole pipeline

    Input:
    csv_path: path to csv file

    Output:
    Prints statistics of prediction accuracy into terminal.
    """
    # 1. Loading data
    try:
        raw_data = load_data(csv_path)
    except FileNotFoundError:
        print(f"Chyba: Soubor '{csv_path}' nebyl nalezen.")
        return

    # 2. Preprocessing
    X, y = data_preprocessing(raw_data)
    print(f"Data po vyčištění: {X.shape}, Features: {list(X.columns)}")

    # 3. Train Test split
    X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=22, stratify=y
    )

    # 4. Imputation
    X_train_processed, X_test_processed = impute_data(X_train, X_test)

    # 5. Final model training
    print("\n--- Trénink finálního modelu ---")
    model = train_final_model(X_train_processed, y_train)

    # 6. Evaluation
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