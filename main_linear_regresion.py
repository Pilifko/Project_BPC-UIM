import numpy as np
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import IterativeImputer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def load_data(
        csv_data
    ) -> DataFrame:
    """
    Parameters
    ----------
    csv_data -> csv input file
    returns -> data from csv in DataFrame
    """
    data = pd.read_csv(csv_data)
    return data

def data_preprocessing(
        data: DataFrame = None
    ) -> tuple:
    """
    Function to preprocess your data.

    Parameters
    ----------
    data -> data to be preprocessed
    df_pca -> output preprocessed data
    target -> output vector of target variable
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

    target = df['target']
    del df_imputed['target']

    df_scaled = StandardScaler().fit_transform(df_imputed)
    df_scaled = pd.DataFrame(df_scaled, columns=df_imputed.columns)

    pca = PCA(n_components=10, svd_solver='full')
    df_pca = pca.fit_transform(df_scaled)
    print(sum(pca.explained_variance_ratio_))
    return df_pca, target

def compute_statistics(y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
    """
    Compute statistics for binary classification.

    Parameters
    ----------
    y_pred : torch.Tensor -> raw logits from model
    y_true : torch.Tensor -> true labels (float, shape [N,1])

    Returns
    -------
    stats : dict -> accuracy, precision, recall, f1, mcc
    """
    # Sigmoid to get probabilities
    y_prob = torch.sigmoid(y_pred).numpy()
    y_class = (y_prob > 0.5).astype(int).flatten()
    y_true_np = y_true.numpy().flatten()

    stats = {
        "accuracy": accuracy_score(y_true_np, y_class),
        "precision": precision_score(y_true_np, y_class),
        "recall": recall_score(y_true_np, y_class),
        "f1": f1_score(y_true_np, y_class),
        "mcc": matthews_corrcoef(y_true_np, y_class)
    }

    return stats

X, y = data_preprocessing(load_data("heart-disease_data.csv"))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=4)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))