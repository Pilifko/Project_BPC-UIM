# -*- coding: utf-8 -*-
# importování modulů:
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import seaborn as sns

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


def process_data(X) -> tuple:
    """
    Imputuje chybějící hodnoty a škáluje data.
    """
    cols = X.columns

    imputer = IterativeImputer(random_state=42, max_iter=10)

    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_imputed)

    X_final = pd.DataFrame(X_scaled, columns=cols)

    return X_final


def visualize_data(X: pd.DataFrame, y: pd.Series):
    """
    Zobrazí korelační matici a rozložení targetu.
    """
    # 1. Korelace
    plt.figure(figsize=(10, 8))
    sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Korelační matice vstupních dat")
    plt.show()

    # 2. Target balance
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title("Rozložení cílové třídy (Target)")
    plt.show()


def plot_learning_curve(model):
    """
    Zobrazí křivku učení (Train vs Test error) pro odhalení přeučení.
    """
    results = model.get_evals_result()

    if 'validation' not in results:
        print("Model neobsahuje validační data pro vykreslení křivky.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(results['learn']['Logloss'], label='Trénovací chyba')
    plt.plot(results['validation']['Logloss'], label='Validační (Test) chyba')
    plt.xlabel('Iterace')
    plt.ylabel('Logloss')
    plt.title('Křivka učení (Learning Curve)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_feature_importance(model, feature_names):
    """
    Zobrazí důležitost jednotlivých příznaků.
    """
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.title('Důležitost příznaků (Feature Importance)')
    plt.show()



def optimize_catboost(X: pd.DataFrame, y: pd.Series, n_trials: int) -> dict:
    """
    Funkce pro Bayesovskou optimalizaci hyperparametrů pomocí Optuny.
    """

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),  # Upraveno pro demo
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'loss_function': 'Logloss',
            'verbose': False,
            'random_seed': 42,
            'allow_writing_files': False
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=43)
        scores = []

        for train_index, val_index in skf.split(X, y):
            X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
            y_tr, y_val = y.iloc[train_index], y.iloc[val_index]

            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)

            preds = model.predict(X_val)
            score = matthews_corrcoef(y_val, preds)
            scores.append(score)

        return np.mean(scores)

    print(f"Spouštím Bayesovskou optimalizaci ({n_trials} pokusů)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print(f"Nejlepší parametry nalezeny: {study.best_params}")
    return study.best_params


def train_final_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict, X_test: pd.DataFrame = None,
                      y_test: pd.Series = None) -> CatBoostClassifier:
    """
    Natrénuje finální model s nejlepšími parametry a vykresluje loss fci.
    """
    full_params = params.copy()
    full_params.update({
        'loss_function': 'Logloss',
        'verbose': 100,
        'random_seed': 42,
        'allow_writing_files': False
    })

    model = CatBoostClassifier(**full_params)

    # Změna: předáváme eval_set pro sledování chyby v průběhu učení
    if X_test is not None and y_test is not None:
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    else:
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

    # --- VIZUALIZACE DAT (EDA) ---
    print("Zobrazuji vizualizaci dat...")
    visualize_data(X, y)
    # -----------------------------

    # 3. Rozdělení na Train a Test (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )

    # 4. Process (Imputace a Škálování)
    X_train_processed = process_data(X_train)
    X_test_processed = process_data(X_test)

    # 5. Bayesovská optimalizace
    best_params = optimize_catboost(X_train_processed, y_train, n_trials=10)

    # 6. Trénink finálního modelu
    print("\n--- Trénink finálního modelu ---")
    # Posíláme i testovací data pro účely grafů (eval_set)
    model = train_final_model(X_train_processed, y_train, best_params, X_test_processed, y_test)

    # --- VIZUALIZACE MODELU ---
    print("Zobrazuji křivky učení a důležitost rysů...")
    plot_learning_curve(model)
    plot_feature_importance(model, X_train_processed.columns)
    # --------------------------

    # 7. Evaluace
    y_pred = model.predict(X_test_processed)
    stats = compute_statistics(y_test, y_pred)

    print("\n--- Výsledky modelu (Test set) ---")
    print(f"Matthews Correlation Coefficient (MCC): {stats['MCC']:.4f}")
    print(f"Accuracy: {stats['Accuracy']:.4f}")
    print(f"F1 Score: {stats['F1 Score']:.4f}")
    print("\nConfusion Matrix:")
    print(stats['Confusion Matrix'])


    y_pred_full = model.predict(process_data(X))
    stats_full = compute_statistics(y, y_pred_full)

    print("\n--- Výsledky modelu (Full dataset) ---")
    print(f"Matthews Correlation Coefficient (MCC): {stats_full['MCC']:.4f}")
    print(f"Accuracy: {stats_full['Accuracy']:.4f}")
    print(f"F1 Score: {stats_full['F1 Score']:.4f}")

    model.save_model('heart_disease_prediction_model')

if __name__ == "__main__":
    main('heart-disease_data.csv')