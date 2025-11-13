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


# My model

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


class NeuralNetwork(nn.Module):
    def __init__(self, in_features=10):
        super(NeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(10, 10)
        self.hidden_layer = nn.Linear(10, 5)
        self.output_layer = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

model = NeuralNetwork()


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

if __name__ == "__main__":
    X, y = data_preprocessing(load_data("heart-disease_data.csv"))
    y = y.to_numpy()

    model = NeuralNetwork()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Převod na tensory
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)

    # Pro BCEWithLogitsLoss: float a shape [N,1]
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    epochs = 200
    losses = []

    for i in range(epochs):
        # Forward pass
        y_pred = model(X_train)

        # Loss
        loss = criterion(y_pred, y_train)

        # Keep track of losses
        losses.append(loss.detach().numpy())

        if i % 10 == 0:
            print(f'Epoch: {i} and loss: {loss.item():.4f}')

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Graph it
    plt.plot(range(epochs), losses)
    plt.ylabel("loss/error")
    plt.xlabel('Epoch')
    plt.show()

    with torch.no_grad():  # Basically turn off back propogation
        y_eval = model.forward(X_test)  # X_test are features from our test set, y_eval will be predictions
        loss = criterion(y_eval, y_test)  # Find the loss or error

    correct = 0
    with torch.no_grad():
        for i, data in enumerate(X_test):
            y_val = model.forward(data)
            if y_val.argmax().item() == y_test[i]:
                correct += 1

    print(f'We got {correct} correct!')

    with torch.no_grad():
        y_eval = model(X_test)  # logits
        loss = criterion(y_eval, y_test)
        print(f'Test loss: {loss.item():.4f}')

        # Spočítat statistiky
        stats = compute_statistics(y_eval, y_test)
        print("Statistics on test set:")
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")