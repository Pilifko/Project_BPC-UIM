# -*- coding: utf-8 -*-


# Import necessary modules
import numpy as np
from main_final import *
import pandas as pd
from catboost import CatBoostClassifier

def test_model(
    csv_file: str = None,
    ) -> np.ndarray:
    """
    Function to test your model.

    Parameters
    ----------
    csv_file -> csv file path (heart disease dataset)
    """

    # Test and return your results
    # load data
    try:
        raw_data = load_data(csv_file)
    except FileNotFoundError:
        print(f"Chyba: Soubor '{csv_file}' nebyl nalezen.")
        return

    X, y = data_preprocessing(raw_data)

    X_processed = process_data(X)

    # load model
    model = CatBoostClassifier()  # parameters not required.
    model.load_model('heart_disease_prediction_model')

    y_pred = model.predict(X_processed)

    stats = compute_statistics(y, y_pred)

    print("\n--- Výsledky modelu (Test set) ---")
    print(f"Matthews Correlation Coefficient (MCC): {stats['MCC']:.4f}")
    print(f"Accuracy: {stats['Accuracy']:.4f}")
    print(f"F1 Score: {stats['F1 Score']:.4f}")
    print("\nConfusion Matrix:")
    print(stats['Confusion Matrix'])


if __name__ == "__main__":
    test_model("heart-diseas_e_data.csv")
