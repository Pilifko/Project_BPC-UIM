from sklearn.model_selection import StratifiedKFold
from main_final_P import *

optuna.logging.set_verbosity(optuna.logging.INFO)

raw_data = load_data("heart-disease_data.csv")
X, y = data_preprocessing(raw_data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=83, stratify=y)


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 700),
        'depth': trial.suggest_int('depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 15),
        'loss_function': 'Logloss',
        'verbose': 200,
        'random_seed': 42,
        'cat_features': ['sex', 'cp', 'restecg', 'thal'],
        'one_hot_max_size': 10

    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)

    scores = []
    for train_index, val_index in skf.split(X_train, y_train):
        X_tr, X_val = impute_data(X_train.iloc[train_index]), impute_data(X_train.iloc[val_index])
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, verbose=False)

        preds = model.predict(X_val)
        score = matthews_corrcoef(y_val, preds)
        scores.append(score)
    score = np.mean(scores)
    return score


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

print(study.best_params)
