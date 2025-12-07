### main_final_P.py

Tento skript slúži primárne na trénovanie a vyhodnotenie finálneho modelu. Pri jeho spustení (pomocou funkcie main) sa dáta spracujú, model natrénuje a uloží ako *heart_disease_prediction_model*.
Tento skript ďalej obsahuje všetky funkcie potrebné pre trénovanie modelu:

* load_data(csv_data): Načítanie surových dát z CSV súboru.

* data_preprocessing(data): Odstráni stĺpce s nízkou koreláciou ku target hodnotám ('fbs','chol') a nahradí nelogické/neplatné hodnoty np.nan.

* impute_data(X, X_test): Imputácia chýbajúcich hodnôt pomocou IterativeImputer (MICE). Táto funkcia tiež volá konečné spracovanie imputovaných dát (process_data).

* process_data(X): Post-imputačné spracovanie. Zaokrúhľuje imputované hodnoty na celé čísla pre kategorické stĺpce ('sex', 'cp', 'restecg', 'thal') a nastavuje správne dátové typy (int).

* train_final_model(X_train, y_train): Trénovanie modelu CatBoostClassifier s vopred definovanými, optimalizovanými hyperparametrami. Tento model sme zvolili pre jeho robustnosť a schopnosť natívne pracovať s kategorickými dátami.

* compute_statistics(y_true, y_pred): Výpočet kľúčových metrík (Accuracy, F1 Score, MCC - Matthews Correlation Coefficient) a zobrazenie Confusion Matrix.

* main(csv_path): Hlavná riadiaca funkcia, ktorá rozdelí dáta, spracuje ich, natrénuje model, vyhodnotí ho a nakoniec uloží model do súboru heart_disease_prediction_model.

### opt.py

Tento skript bol použitý na automatickú optimalizáciu hyperparametrov modelu CatBoost pomocou knižnice Optuna.

* Skript načíta a predpripraví dáta pomocou funkcií z main_final_P.py, rozdelí ich na trénovaciu a testovaciu sadu.

* objective(trial): Funkcia definuje premenné pre optimalizáciu. Pre každý trial navrhne Optuna sadu hyperparametrov (iterations, depth, learning_rate, l2_leaf_reg).

  * Na testovacej sade je použitá funkcia Stratified K-Fold Cross-Validation (s n_splits=5)

  * Model je natrénovaný s navrhnutými parametrami a jeho výkon je hodnotený pomocou Matthews Correlation Coefficient (MCC). Optuna sa snaží maximalizovať túto metriku (direction='maximize').

* Výstup: Po dokončení stanoveného počtu skúšok (n_trials) vypíše najlepšiu nájdenú sadu parametrov (study.best_params).

### testing_P.py

Tento skript slúži na testovanie a validáciu už uloženého modelu.

* run_model(csv_file): Hlavná funkcia skriptu.

  * Načíta dáta z určeného CSV súboru.

  * Spracuje dáta pomocou preddefinovaných krokov (data_preprocessing a impute_data).

  * Načíta uložený model *heart_disease_prediction_model*.

  * Vykoná predikcie na spracovaných dátach a vypočíta finálne štatistiky (MCC, Accuracy, F1 Score a Confusion Matrix).