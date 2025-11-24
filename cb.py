import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool  # Import Pool
from imblearn.over_sampling import SMOTENC  # Still imported, but not used in the pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification  # Used for synthetic data example
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix  # Added missing imports
import matplotlib.pyplot as plt  # Import for plotting
import seaborn as sns  # Import for advanced plotting

# --- 1. DATA LOADING (Simulated) ---

# Define the feature names and indices based on the common Heart Disease dataset order
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Indices used in your original script (must match FEATURE_NAMES)
CAT_INDICES_ORIGINAL = [1, 2, 5, 6, 8, 10, 11, 12]  # e.g., sex, cp, fbs, etc.
NUM_INDICES_ORIGINAL = [0, 3, 4, 7, 9]  # e.g., age, trestbps, chol, etc.


def load_data(csv_path: str):
    """
    Loads data and separates features (X) and target (y).
    NOTE: Using synthetic data for demonstration as the CSV is unavailable.
    """
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File '{csv_path}' not found. Generating synthetic data instead.")
        # Generate synthetic data with the expected 13 features
        X_synth, y_synth = make_classification(
            n_samples=250, n_features=13, n_informative=8, n_classes=2,
            weights=[0.7, 0.3], random_state=42
        )
        X_synth = pd.DataFrame(X_synth, columns=FEATURE_NAMES)

        # Ensure categorical columns are int/float for CatBoost
        for i in CAT_INDICES_ORIGINAL:
            X_synth.iloc[:, i] = X_synth.iloc[:, i].apply(lambda x: 1 if x > 0.5 else 0).astype(float)

        return X_synth, y_synth

    # If loading succeeds:
    y = data['target']
    X = data.drop(columns=['target'])
    return X, y


# --- 2. STATELESS VALIDATION FUNCTION ---

def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Performs range checks and converts invalid values to NaN.
    This is a stateless operation and can be applied globally or inside a FunctionTransformer.
    """
    df = data.copy()

    # 1. Age: 0-120
    df['age'] = df['age'].apply(lambda x: x if 0 <= x <= 120 else np.nan).astype(float)
    # 2. Categorical features checks (only check for NaNs, as they should be integers)
    # This assumes integer categories (0, 1, 2, 3, etc.)
    for col, valid_values in [
        ('sex', [0, 1]), ('cp', [0, 1, 2, 3]), ('fbs', [0, 1]), ('restecg', [0, 1, 2]),
        ('exang', [0, 1]), ('slope', [0, 1, 2]), ('ca', [0, 1, 2, 3]), ('thal', [0, 1, 2, 3])
    ]:
        df[col] = df[col].apply(lambda x: x if x in valid_values else np.nan).astype(float)

    # 3. Continuous features checks
    df['trestbps'] = df['trestbps'].apply(lambda x: x if 100 <= x <= 300 else np.nan).astype(float)
    df['chol'] = df['chol'].apply(lambda x: x if 50 <= x <= 300 else np.nan).astype(float)
    df['thalach'] = df['thalach'].apply(lambda x: x if 50 <= x <= 250 else np.nan).astype(float)
    df['oldpeak'] = df['oldpeak'].apply(lambda x: x if 0 <= x <= 3 else np.nan).astype(float)

    # Note: Returning the validated DataFrame with NaNs (Imputation is done later in the pipeline)
    return df


# --- 3. PIPELINE COMPONENTS ---

# CRITICAL: Define the ColumnTransformer to handle imputation STATEFULLY
preprocessor = ColumnTransformer(
    transformers=[
        # Numeric Features Pipeline: ONLY Impute NaNs (Scaler removed)
        ('num_pipe', Pipeline([
            ('iter_imputer', IterativeImputer(random_state=0)),  # STATEFUL: Fitted per fold
        ]), NUM_INDICES_ORIGINAL),

        # Categorical Features Pipeline: Impute NaNs, then Passthrough
        ('cat_pipe', Pipeline([
            ('simple_imputer', SimpleImputer(strategy='most_frequent')),  # STATEFUL: Fitted per fold
            ('passthrough', 'passthrough')
        ]), CAT_INDICES_ORIGINAL)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

# Recalculate categorical indices after ColumnTransformer runs:
CAT_INDICES_AFTER_CT = list(range(len(NUM_INDICES_ORIGINAL), len(FEATURE_NAMES)))
NUM_INDICES_AFTER_CT = list(range(len(NUM_INDICES_ORIGINAL)))


# --- FUNCTION FOR TYPE CONVERSION ---
def convert_to_int_catboost(X):
    """
    Converts the categorical columns to integer type and the output back to a DataFrame.
    This is necessary because Imputers return floats.
    """
    # 1. Convert NumPy array back to DataFrame for better type handling
    X_df = pd.DataFrame(X, columns=FEATURE_NAMES)

    # 2. Iterate over the known names of categorical features
    for i in CAT_INDICES_AFTER_CT:
        col_name = FEATURE_NAMES[i]

        # Round and cast to int (Imputer output is float)
        X_df[col_name] = np.round(X_df[col_name]).astype(int)

    # Ensure numerical columns remain floats (they are already scaled)
    for i in NUM_INDICES_AFTER_CT:
        col_name = FEATURE_NAMES[i]
        X_df[col_name] = X_df[col_name].astype(float)

    return X_df


# Define CatBoost model parameters
model = CatBoostClassifier(
    iterations=400,
    learning_rate=0.02,
    depth=3,
    loss_function='Logloss',
    eval_metric='F1',
    verbose=False,
    random_state=39,
    # CRITICAL CHANGE: Use native class weighting instead of SMOTENC
    auto_class_weights='Balanced',
    cat_features=CAT_INDICES_AFTER_CT
)

# --- 4. THE FINAL LEAK-FREE IMBLEARN PIPELINE ---

# NOTE: ImbPipeline is used but sampling is removed, so it functions as a standard Pipeline.
smotenc_pipeline = ImbPipeline([
    # Step 1: Stateless Validation
    ('stateless_validation', FunctionTransformer(func=validate_data, validate=False)),

    # Step 2: Stateful Preprocessing (ONLY Imputation, fitted per fold)
    ('preprocessor', preprocessor),

    # Step 3: Type Conversion (Still needed because Imputers output floats)
    ('type_conversion', FunctionTransformer(func=convert_to_int_catboost, validate=False)),

    # Step 4: Model Training (now handles imbalance internally)
    ('model', model)
])

# --- 5. EXECUTION AND VISUALIZATION ---

# Load Data and Split into Train/Test sets
X, y = load_data("heart-disease_data.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("--- Cross-Validation Performance (Reliable Estimate) ---")
# cross_val_score executes the full leak-free pipeline 5 times
scores = cross_val_score(smotenc_pipeline, X_train, y_train, cv=kfold, scoring='f1')
mcc_scores = cross_val_score(smotenc_pipeline, X_train, y_train, cv=kfold, scoring='matthews_corrcoef')

print(f"MCC Scores per fold: {mcc_scores}")
print(f"Mean F1 Score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
print(f"Mean MCC Score: {np.mean(mcc_scores):.4f} (+/- {np.std(mcc_scores):.4f})")

print("\n--- Final Model Training and Test Set Evaluation ---")

# 1. Fit the final pipeline on the ENTIRE training set (X_train)
smotenc_pipeline.fit(X_train, y_train)

# 2. Predict on the completely unseen test set (X_test)
y_pred_test = smotenc_pipeline.predict(X_test)

print("Final Test Set Metrics:")
print(f"Accuracy : {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"MCC      : {matthews_corrcoef(y_test, y_pred_test):.4f}")
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))

# --- 6. VISUALIZATION OF TOP FEATURES AND PREDICTIONS (FIXED) ---

# Get the final CatBoost model from the pipeline
final_model = smotenc_pipeline.named_steps['model']

# Create a transformer pipeline containing all preprocessing steps (excluding the model)
# This is used to ensure X_train has the correct types and imputed values for the Pool object.
transformer_steps = smotenc_pipeline.steps[:-1]
data_transformer = Pipeline(transformer_steps)

# Transform X_train using the fitted preprocessors
# X_train_transformed is now a DataFrame with the correct int/float dtypes
X_train_transformed = data_transformer.transform(X_train)

# 6.1. Extract Feature Importance (FIXED: Using the transformed data)
# The transformed data (DataFrame with correct dtypes) must be used for the Pool
feature_importances = final_model.get_feature_importance(Pool(
    X_train_transformed,  # Use the correctly transformed DataFrame
    y_train,
    cat_features=CAT_INDICES_AFTER_CT
))
feature_names = X_train_transformed.columns.tolist()  # Use names from the transformed DataFrame

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

TOP_FEATURES = importance_df['Feature'].head(3).tolist()
print("\nTop 3 Important Features for Visualization:", TOP_FEATURES)

# 6.2. Generate CV Prediction Probabilities (on X_train)
# cross_val_predict works fine as it correctly executes the pipeline steps internally.
y_proba_cv = cross_val_predict(
    smotenc_pipeline,
    X_train,
    y_train,
    cv=kfold,
    method='predict_proba'
)[:, 1]  # Probability of the positive class (class 1)

# 6.3. Prepare Data for Plotting
plot_df = X_train_transformed.copy()  # Start with the fully preprocessed, type-corrected features
plot_df['True Class'] = y_train
plot_df['Pred Proba (CV)'] = y_proba_cv
plot_df['Pred Class (CV)'] = (y_proba_cv > 0.5).astype(int)

# Create a combined class column to see True Class vs Misclassification
plot_df['Class_Label'] = plot_df.apply(
    lambda row: f'Class {int(row["True Class"])} (TP/TN)'
    if row['True Class'] == row['Pred Class (CV)'] else
    f'Misclassified as {int(row["Pred Class (CV)"])}',
    axis=1
)

# 6.4. Generate Matrix Scatter Plot (Pair Plot)
plot_features = TOP_FEATURES + ['Pred Proba (CV)']

plt.figure(figsize=(12, 12))
# The hue shows the True Class, allowing us to see how features separate classes.
# The diagonal plots the distribution of each feature.
# The scatter plots show relationships between feature pairs.
sns.pairplot(
    plot_df,
    vars=plot_features,
    hue='Class_Label',  # Use the combined column to highlight correct/incorrect predictions
    palette={'Class 1 (TP/TN)': 'green',
             'Class 0 (TP/TN)': 'blue',
             'Misclassified as 0': 'red',  # False Negative (Missed Positive)
             'Misclassified as 1': 'orange'},  # False Positive (False Alarm)
    diag_kind='kde',
    markers=["o", "s", "D", "X"]
)
plt.suptitle(
    f"Matrix Scatter Plot of Top 3 Features vs. CV Prediction Probability\n(Colored by True Class and Misclassification)",
    y=1.02)
plt.show()

# 6.5. Generate a Feature Importance Bar Plot for context
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title('Top 10 CatBoost Feature Importances')
plt.show()

...