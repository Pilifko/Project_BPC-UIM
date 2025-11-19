import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from main import *

X, y = data_preprocessing(load_data("heart-disease_data.csv"))

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 1200),
        'depth': trial.suggest_int('depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'loss_function': 'Logloss',
        'verbose': False
    }

    model = CatBoostClassifier(**params)
    score = cross_val_score(model, X, y, cv=5, scoring='f1').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(study.best_params)