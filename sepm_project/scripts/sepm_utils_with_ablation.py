
from typing import Dict, Any, Union

import numpy as np
import pandas as pd
import shap
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import t



# %%
def tune_hyperparameters(train_features: np.ndarray, train_target: np.ndarray, model: Any, param_grid: Dict[str, Any],
                         n_inner_splits: int, random_state: int = 1,
                         sample_weights: np.ndarray = None) -> Any:
    """Perform hyperparameter tuning using grid search.

    Parameters:
        train_features (np.ndarray): Training data features.
        train_target (np.ndarray): Training data labels.
        model (Any): Classifier.
        param_grid (Dict[str, Any]): Hyperparameter grid.
        n_inner_splits (int): Number of cross-validation splits for inner fold.
        random_state (int): Random state for reproducibility. Defaults to 1.
        sample_weights (np.ndarray, optional): Sample weights to be used during fitting. Defaults to None.

    Returns:
        Any: Best estimator from the grid search.
    """
    # Validate inputs
    if train_features is None or train_target is None or model is None or param_grid is None:
        raise ValueError("Input parameters cannot be null")

    if not isinstance(n_inner_splits, int) or n_inner_splits <= 0:
        raise ValueError("n_inner_splits must be a positive integer")

    if not isinstance(random_state, int):
        raise ValueError("random_state must be an integer")

    if sample_weights is not None and sample_weights.shape[0] != train_features.shape[0]:
        raise ValueError("Sample weights must have the same number of elements as train_features")

    # Create GridSearchCV instance
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state),
        scoring='f1',
        n_jobs=-1
    )

    # Fit the grid search to the data
    grid_search.fit(train_features, train_target, sample_weight=sample_weights)

    # Extract and return the best estimator
    if not hasattr(grid_search, 'best_estimator_'):
        raise AttributeError("GridSearchCV instance must have best_estimator_ attribute")

    return grid_search.best_estimator_


# %%
def calculate_mean_abs_shap_values(model: Any, train_features: np.ndarray, model_name: str,
                                   standardize_shap: bool = False,
                                   mean_shap: bool = True) -> np.ndarray:
    """
    Calculate SHAP values for a given model.

    Parameters:
        model (Union[shap.Explainer, Any]): Trained model.
        train_features (np.ndarray): Training data.
        model_name (str): Model name.
        standardize_shap (bool): Whether to standardize SHAP values before calculating the mean absolute values.
        mean_shap (bool): Whether to calculate the mean SHAP values for each instance.

    Returns:
        np.ndarray: Mean SHAP values for each instance.
    """

    # Input validation
    if not isinstance(train_features, np.ndarray) or train_features.ndim != 2:
        raise ValueError("train_features must be a 2D numpy array.")

    if model_name not in ["SVM", "Logistic Regression", "XGBoost", "Random Forest", "GBM"]:
        raise ValueError(
            "Invalid model_name. Choose from 'SVM', 'Logistic Regression', 'XGBoost', 'Random Forest', or 'GBM'.")

    # Create explainer based on model name
    if model_name == "SVM":
        explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(train_features, 5))
    elif model_name == "Logistic Regression":
        explainer = shap.LinearExplainer(model, train_features)
    else:
        explainer = shap.TreeExplainer(model)

    try:
        shap_values = explainer.shap_values(train_features)
    except Exception as e:
        raise RuntimeError(f"Error calculating SHAP values: {e}")

    # Handle a case where shap_values returns a list (e.g., for classification)
    if isinstance(shap_values, list):
        if len(shap_values) < 2:
            raise ValueError("SHAP values list does not contain expected class values.")
        shap_values = shap_values[1]  # Take the SHAP values for the positive class

    # For multiclass classification, reduce to the SHAP values for the target class
    if shap_values.ndim == 3:
        if shap_values.shape[2] < 2:
            raise ValueError("Expected at least 2 classes for SHAP values.")
        shap_values = shap_values[:, :, 1]

    # Optionally standardize SHAP values
    if standardize_shap:
        shap_values = StandardScaler().fit_transform(shap_values)

    # Optionally calculate the mean SHAP values for each instance
    if mean_shap:
        shap_values = np.mean(np.abs(shap_values), axis=1).reshape(-1, 1)

    return shap_values


def preprocess_data_with_imputation(features: np.ndarray, target: np.ndarray, test_size: float, standardize: bool,
                                    stratify: np.ndarray, random_state: int):
    """
    Preprocess the data: split into train/test sets, handle missing values using a Random Forest model,
    and standardize if required.

    Args:
    - features (np.ndarray): Feature matrix.
    - target (np.ndarray): Target vector.
    - test_size (float): Proportion of data to include in the test split (between 0 and 1).
    - standardize (bool): Whether to standardize the data.
    - stratify (np.ndarray): Target vector for stratification.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - train_features (np.ndarray): Preprocessed training features.
    - test_features (np.ndarray): Preprocessed testing features.
    - train_target (np.ndarray): Training labels.
    - test_target (np.ndarray): Testing labels.
    """

    # Validate inputs
    if not isinstance(features, np.ndarray) or not isinstance(target, np.ndarray):
        raise ValueError("X and y must be numpy arrays.")
    if not (0 < test_size < 1):
        raise ValueError("test_size must be a float between 0 and 1.")
    if not isinstance(standardize, bool):
        raise ValueError("standardize must be a boolean value.")
    if random_state is not None and not isinstance(random_state, int):
        raise ValueError("random_state must be an integer or None.")
    if len(np.unique(target)) < 2:
        raise ValueError("y must contain at least two classes for stratification.")

    # Check for missing values
    missing_cols = np.where(np.isnan(features).any(axis=0))[0]
    has_missing = len(missing_cols) > 0

    # Split data into training and testing sets
    try:
        train_features, test_features, train_target, test_target = train_test_split(
            features, target, test_size=test_size, random_state=random_state, stratify=stratify
        )

        if has_missing:
            print(f"Imputing missing values for columns: {missing_cols.tolist()}")

            # Initialize IterativeImputer
            imputer = IterativeImputer(
                estimator=LinearRegression(),
                max_iter=5000,
                random_state=random_state
            )
            # Impute missing values
            train_features = imputer.fit_transform(train_features)
            test_features = imputer.transform(test_features)
        else:
            print("No missing values found. Proceeding without imputation.")

        # Standardize data if required
        if standardize:
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_features)
            test_features = scaler.transform(test_features)

        return train_features, test_features, train_target, test_target

    except Exception as e:
        raise RuntimeError(f"An error occurred during data preprocessing: {e}")



# %%
def tune_and_calculate_shap(model, param_grid, X_train, y_train, model_name, inner_splits, random_state,
                            standardize_shap):
    """
    Tune hyperparameters and calculate mean absolute SHAP values, recording SHAP computation time.

    Returns
    -------
    best_model : estimator
        Best estimator found by inner CV.
    mean_abs_shap_values : np.ndarray
        Mean absolute SHAP per training instance (shape: [n_train]).
    shap_time : float
        Wall-clock time (seconds) spent computing SHAP values.
    """
    import time

    best_model = tune_hyperparameters(X_train, y_train, model, param_grid, inner_splits, random_state)

    t0 = time.time()
    mean_abs_shap_values = calculate_mean_abs_shap_values(best_model, X_train, model_name, standardize_shap)
    shap_time = time.time() - t0

    return best_model, mean_abs_shap_values, shap_time



def summarize_with_ci_best_per_fold(df: pd.DataFrame, ci: float = 0.95) -> pd.DataFrame:
    required = {'Model', 'Method', 'Fold', 'F1 Score', 'Balanced Accuracy', 'Accuracy Score'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # metrics we’ll summarize if present
    metrics = ['F1 Score', 'Balanced Accuracy', 'Accuracy Score']
    for extra in ['Brier Score', 'ECE']:
        if extra in df.columns:
            metrics.append(extra)

    def pick_best(g):
        # choose max F1; tie -> smaller Top Percentage
        max_f1 = g['F1 Score'].max()
        cand = g[g['F1 Score'] == max_f1]
        if 'Top Percentage' in cand.columns:
            cand = cand.sort_values('Top Percentage', ascending=True)
        return cand.iloc[[0]]

    # ---- best-per-fold selection ----
    best_rows = []
    for (model, fold), g in df.groupby(['Model', 'Fold']):
        tml = g[g['Method'] == 'TML']
        if not tml.empty:
            best_rows.append(tml.iloc[[0]])

        sepm = g[g['Method'] == 'SEPM']
        if not sepm.empty:
            best_rows.append(pick_best(sepm))

        sepmr = g[g['Method'] == 'SEPM-RAND']
        if not sepmr.empty:
            best_rows.append(pick_best(sepmr))

    best_df = pd.concat(best_rows, ignore_index=True)

    # ---- CI across folds ----
    alpha = 1 - ci
    out_rows = []

    for (method, model), g in best_df.groupby(['Method', 'Model']):
        n = g['Fold'].nunique()
        dfree = n - 1
        tcrit = t.ppf(1 - alpha/2, dfree) if dfree > 0 else np.nan

        row = {'Method': method, 'Model': model, 'N Folds': n}

        for m in metrics:
            x = g[m].astype(float).dropna().values
            nn = len(x)
            mean = np.mean(x) if nn else np.nan
            sd = np.std(x, ddof=1) if nn > 1 else np.nan
            se = sd / np.sqrt(nn) if nn > 1 else np.nan
            half = tcrit * se if nn > 1 else np.nan

            row[f"Mean {m}"] = mean
            row[f"SD {m}"] = sd
            row[f"CI Low {m}"] = mean - half if nn > 1 else np.nan
            row[f"CI High {m}"] = mean + half if nn > 1 else np.nan

        if 'Top Percentage' in g.columns:
            row["Mean Top Percentage"] = g['Top Percentage'].mean()
            row["SD Top Percentage"] = g['Top Percentage'].std(ddof=1)

        out_rows.append(row)

    return pd.DataFrame(out_rows)
