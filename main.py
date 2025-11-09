# -*- coding: utf-8 -*-

"""
Created on 11. 09. 2025 at 11:13:56

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Tento script slouží jako hlavní spouštěcí bod pro projekt.
    Skript berte jako volný rámec, který můžete upravit dle svých potřeb.    
"""
# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.impute import IterativeImputer


# My model

def load_data(
        csv_data
    ) -> np.ndarray:
    """
    Parameters
    ----------
    csv_data -> csv input file
    returns -> data from csv in ndarray
    """
    data = pd.read_csv(csv_data)
    return data


def data_preprocessing(
        data: np.ndarray = None
    ) -> np.ndarray:
    """
    Function to preprocess your data.

    Parameters
    ----------
    data -> data to be preprocessed
    df_imputed -> output preprocessed data
    """
    # Preprocess and return your data
    df = pd.DataFrame(data)

    df['age'] = df['age'].apply(lambda x: int(x) if 0 <= x <= 120 else np.nan).astype(float)
    df['sex'] = df['sex'].apply(lambda x: x if x in [0, 1] else np.nan).astype(float)
    df['cp'] = df['cp'].apply(lambda x: x if x in [0, 1, 2, 3] else np.nan).astype(float)
    df['trestbps'] = df['trestbps'].apply(lambda x: int(x) if 100 <= x <= 300 else np.nan).astype(float)
    df['chol'] = df['chol'].apply(lambda x: int(x) if 50 <= x <= 300 else np.nan).astype(float)
    df['fbs'] = df['fbs'].apply(lambda x: x if x in [0, 1] else np.nan).astype(float)
    df['restecg'] = df['restecg'].apply(lambda x: x if x in [0, 1, 2] else np.nan).astype(float)
    df['thalach'] = df['thalach'].apply(lambda x: int(x) if 50 <= x <= 250 else np.nan).astype(float)
    df['exang'] = df['exang'].apply(lambda x: x if x in [0, 1] else np.nan).astype(float)
    df['oldpeak'] = df['oldpeak'].apply(lambda x: int(x) if 0 <= x <= 3 else np.nan).astype(float)
    df['slope'] = df['slope'].apply(lambda x: x if x in [0, 1, 2] else np.nan).astype(float)
    df['ca'] = df['ca'].apply(lambda x: x if x in [0, 1, 2, 3] else np.nan).astype(float)
    df['thal'] = df['thal'].apply(lambda x: x if x in [0, 1, 2, 3] else np.nan).astype(float)
    df['target'] = df['target'].apply(lambda x: x if x in [0, 1] else np.nan).astype(float)

    imputer = IterativeImputer(random_state=0)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    df_imputed = round(df_imputed)
    return df_imputed

def my_model(
    # Add your parameters here....
    ) -> np.ndarray:
    """
    Class representing your model.
    """
    raise NotImplementedError("Function my_model() is not implemented.")

def main(
    # Add your parameters here....
    ) -> None:
    """
    Main function to run the project.

    Parameters
    ----------
    Add your parameters here....
    """
    # Initialize and run your model

    # model = my_model(
    #     # Pass your parameters here....
    # )

    # Add your code here....

    # Print Matthews Correlation Coefficient (MCC)
    # print(f"Matthews Correlation Coefficient (MCC): {matthews_corrcoef}")
    raise NotImplementedError("Function main() is not implemented.")

def compute_statistics(
    # Add your parameters here....
    # model_output: np.ndarray = None,
    # ground_truth: np.ndarray = None
    ) -> tuple:
    """
    Function to compute statistics.

    Parameters
    ----------
    Add your parameters here....
    """
    # Compute Matthews Correlation Coefficient (MCC)
    # matthews_corrcoef = None
    raise NotImplementedError("Function compute_statistics() is not implemented.")

if __name__ == "__main__":
    main(
        # Pass your parameters here....
    )
