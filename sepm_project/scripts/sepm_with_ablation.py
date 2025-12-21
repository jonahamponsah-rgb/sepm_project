import gc
import warnings
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from scipy.special import expit

from sepm_project.scripts.sepm_utils_with_ablation import (
    tune_hyperparameters,
    tune_and_calculate_shap,
    preprocess_data_with_imputation
)

warnings.filterwarnings('ignore')


def expected_calibration_error(y_true, y_prob, n_bins=10, strategy="uniform"):
    """
    Expected Calibration Error (ECE) for binary classification.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0/1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    n_bins : int
        Number of bins.
    strategy : {"uniform", "quantile"}
        Binning strategy.

    Returns
    -------
    float
        ECE (lower is better).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)
    elif strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(y_prob, quantiles)
        bin_edges[0], bin_edges[-1] = 0.0, 1.0
        bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
    return float(ece)


def train_test_split_evaluation(
    X, y, test_size, inner_splits, random_state,
    standardize, standardize_shap,
    model_configs, top_percentages
):
    """
    Evaluate models using a single train-test split with SEPM and TML.
    """

    # Append 1.0 to top_percentage if it is not already in the list
    if top_percentages is None:
        top_percentages = [0.5, 1.0]
    elif 1.0 not in top_percentages:
        top_percentages.append(1.0)

    # Preprocess and split data
    X_train, X_test, y_train, y_test = preprocess_data_with_imputation(
        X, y, test_size, standardize, y, random_state
    )

    results = []
    for model_config in model_configs:
        model_name = model_config['name']
        model = model_config['classifier']
        param_grid = model_config['param_grid']

        best_model, mean_abs_shap_values, shap_time = tune_and_calculate_shap(
            model, param_grid, X_train, y_train, model_name,
            inner_splits=inner_splits, random_state=random_state,
            standardize_shap=standardize_shap
        )

        # Perform TML method
        y_pred_tml = best_model.predict(X_test)
        tml_result = {
            'Model': model_name,
            'Method': 'TML',
            'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred_tml),
            'F1 Score': f1_score(y_test, y_pred_tml, zero_division=0)
        }

        def process_sepm(top_percentage):
            """
            SEPM includes TML when top_percentage = 1.
            """
            shap_threshold = np.percentile(mean_abs_shap_values, 100 * (1 - top_percentage))
            top_indices = np.where(mean_abs_shap_values >= shap_threshold)[0]

            # Ensure both classes are present in the selected subset.
            if len(np.unique(y_train[top_indices])) < 2:
                missing_classes = set(np.unique(y_train)) - set(np.unique(y_train[top_indices]))
                for missing_class in missing_classes:
                    candidate_indices = np.where(y_train == missing_class)[0]
                    candidate_indices = np.setdiff1d(candidate_indices, top_indices)
                    if candidate_indices.size > 0:
                        extra_index = candidate_indices[np.argmax(mean_abs_shap_values[candidate_indices])]
                        top_indices = np.append(top_indices, extra_index)
                    else:
                        top_indices = np.arange(len(y_train))

            X_train_sepm, y_train_sepm = X_train[top_indices], y_train[top_indices]

            best_model_sepm = tune_hyperparameters(
                X_train_sepm, y_train_sepm, model, param_grid,
                inner_splits, random_state
            )
            y_pred_sepm = best_model_sepm.predict(X_test)

            return {
                'Model': model_name,
                'Method': 'SEPM',
                'Top Percentage': top_percentage,
                'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred_sepm),
                'F1 Score': f1_score(y_test, y_pred_sepm, zero_division=0),
                'Shap Time': shap_time
            }

        # Run SEPM methods
        sepm_results = Parallel(n_jobs=-1)(
            delayed(process_sepm)(top_percentage) for top_percentage in top_percentages
        )

        results.extend([tml_result] + sepm_results)

    all_results_df = pd.DataFrame(results)

    best_scores = all_results_df.groupby(['Method', 'Model']).agg(
        best_f1=('F1 Score', 'max'),
        best_balanced_accuracy=('Balanced Accuracy', 'max'),
    ).reset_index()

    best_scores.columns = ['Method', 'Model', 'F1 Score', 'Balanced Accuracy']
    return best_scores


def nested_cross_validation(
    X, y, model_config,
    outer_splits, inner_splits, random_state,
    standardize, standardize_shap,
    top_percentages
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Nested cross-validation integrating:
      - TML: train on all training data in each outer fold
      - SEPM: SHAP-ranked patient subset selection across a grid of top_percentages
      - SEPM-RAND: ablation control using stratified random subsampling
                  of the SAME subset size as SEPM for each top_percentage

    Returns
    -------
    results_df : pd.DataFrame
        Full per-fold metrics for all evaluated top_percentages.
    best_fold_predictions_df : pd.DataFrame
        Out-of-fold predictions for ONLY:
          - TML (one per fold)
          - SEPM (best-p per fold, chosen by max F1; tie -> smaller p)
          - SEPM-RAND (best-p per fold, chosen by max F1; tie -> smaller p)
        Stored as arrays in columns: y_true, y_prob
    """

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)

    X = np.asarray(X)
    y = np.asarray(y)
    if np.isnan(X).any():
        imputer = IterativeImputer(estimator=LinearRegression(), max_iter=500, random_state=random_state)
        X = imputer.fit_transform(X)

    model_name = model_config["name"]
    model = model_config["classifier"]
    param_grid = model_config["param_grid"]

    if not top_percentages:
        top_percentages = [1.0]

    def _get_probs(estimator, X_test):
        """Return P(y=1). Falls back to sigmoid(decision_function) if needed."""
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(X_test)[:, 1]
        return expit(estimator.decision_function(X_test))

    def _proba_metrics(estimator, X_test, y_test):
        if hasattr(estimator, "predict_proba") or hasattr(estimator, "decision_function"):
            y_prob = _get_probs(estimator, X_test)
            brier = brier_score_loss(y_test, y_prob)
            ece = expected_calibration_error(y_test, y_prob, n_bins=10, strategy="uniform")
            return brier, ece
        return np.nan, np.nan

    def _stratified_random_subset(y_train, n_subset, rng):
        """Stratified random subset of size n_subset (try to keep >=1 per class when feasible)."""
        n_total = len(y_train)
        if n_subset >= n_total:
            return np.arange(n_total)

        y_train = np.asarray(y_train)
        classes, counts = np.unique(y_train, return_counts=True)

        # proportional allocation (min 1 per class)
        target = {c: max(1, int(round(n_subset * (cnt / n_total)))) for c, cnt in zip(classes, counts)}
        current = sum(target.values())

        # shrink if too large (keep >=1 per class)
        while current > n_subset:
            c = max(target, key=lambda k: target[k])
            if target[c] > 1:
                target[c] -= 1
                current -= 1
            else:
                break

        # grow if too small
        while current < n_subset:
            c = max(target, key=lambda k: (counts[np.where(classes == k)[0][0]] - target[k]))
            target[c] += 1
            current += 1

        chosen = []
        for c in classes:
            idx_c = np.where(y_train == c)[0]
            k = min(target[c], len(idx_c))
            if k > 0:
                chosen.extend(rng.choice(idx_c, size=k, replace=False).tolist())

        chosen = np.array(chosen, dtype=int)

        if chosen.size < n_subset:
            remaining = np.setdiff1d(np.arange(n_total), chosen)
            fill = rng.choice(remaining, size=(n_subset - chosen.size), replace=False)
            chosen = np.concatenate([chosen, fill])
        elif chosen.size > n_subset:
            chosen = rng.choice(chosen, size=n_subset, replace=False)

        return chosen

    results = []
    best_fold_predictions = []  # <-- ONLY best-p per fold (TML always included)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize within fold
        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Tune full-data model + compute SHAP once per fold
        best_model_full, mean_abs_shap_values, shap_time = tune_and_calculate_shap(
            model, param_grid, X_train, y_train,
            model_name, inner_splits, random_state, standardize_shap
        )

        # ---------- TML ----------
        y_pred_tml = best_model_full.predict(X_test)
        brier_tml, ece_tml = _proba_metrics(best_model_full, X_test, y_test)

        results.append({
            "Fold": fold_idx,
            "Model": model_name,
            "Method": "TML",
            "Top Percentage": 1.0,
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred_tml),
            "F1 Score": f1_score(y_test, y_pred_tml, zero_division=0),
            "Accuracy Score": accuracy_score(y_test, y_pred_tml),
            "Brier Score": brier_tml,
            "ECE": ece_tml,
            "Shap Time": shap_time,
        })

        # Store TML predictions (one per fold)
        y_prob_tml = _get_probs(best_model_full, X_test)
        best_fold_predictions.append({
            "Fold": fold_idx,
            "Model": model_name,
            "Method": "TML",
            "Top Percentage": 1.0,
            "y_true": y_test.astype(int),
            "y_prob": y_prob_tml.astype(float),
        })

        # Track best-p per fold (matches your table rule)
        best_sepm_pred_record: Optional[Dict[str, Any]] = None
        best_sepm_f1 = -np.inf
        best_sepm_p = np.inf

        best_rand_pred_record: Optional[Dict[str, Any]] = None
        best_rand_f1 = -np.inf
        best_rand_p = np.inf

        # ---------- SEPM + SEPM-RAND ----------
        for top_percentage in top_percentages:
            # SHAP threshold and indices
            shap_threshold = np.percentile(mean_abs_shap_values, 100 * (1 - top_percentage))
            top_indices = np.where(mean_abs_shap_values >= shap_threshold)[0]

            # Ensure both classes present; if not, add one high-SHAP sample from missing class
            if len(np.unique(y_train[top_indices])) < 2:
                missing_classes = set(np.unique(y_train)) - set(np.unique(y_train[top_indices]))
                for missing_class in missing_classes:
                    candidate_indices = np.where(y_train == missing_class)[0]
                    candidate_indices = np.setdiff1d(candidate_indices, top_indices)
                    if candidate_indices.size > 0:
                        extra_index = candidate_indices[np.argmax(mean_abs_shap_values[candidate_indices])]
                        top_indices = np.append(top_indices, extra_index)
                    else:
                        top_indices = np.arange(len(y_train))

            p_val = float(top_percentage)

            # ---- SEPM (SHAP-ranked) ----
            X_train_sepm, y_train_sepm = X_train[top_indices], y_train[top_indices]
            best_model_sepm = tune_hyperparameters(
                X_train_sepm, y_train_sepm, model, param_grid, inner_splits, random_state
            )
            y_pred_sepm = best_model_sepm.predict(X_test)
            brier_sepm, ece_sepm = _proba_metrics(best_model_sepm, X_test, y_test)

            f1_sepm = f1_score(y_test, y_pred_sepm, zero_division=0)

            results.append({
                "Fold": fold_idx,
                "Model": model_name,
                "Method": "SEPM",
                "Top Percentage": p_val,
                "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred_sepm),
                "F1 Score": f1_sepm,
                "Accuracy Score": accuracy_score(y_test, y_pred_sepm),
                "Brier Score": brier_sepm,
                "ECE": ece_sepm,
                "Shap Time": shap_time,
            })

            # Update best SEPM per fold (tie -> smaller p)
            y_prob_sepm = _get_probs(best_model_sepm, X_test)
            if (f1_sepm > best_sepm_f1) or (np.isclose(f1_sepm, best_sepm_f1) and p_val < best_sepm_p):
                best_sepm_f1 = f1_sepm
                best_sepm_p = p_val
                best_sepm_pred_record = {
                    "Fold": fold_idx,
                    "Model": model_name,
                    "Method": "SEPM",
                    "Top Percentage": p_val,
                    "y_true": y_test.astype(int),
                    "y_prob": y_prob_sepm.astype(float),
                }

            # ---- SEPM-RAND (ablation) ----
            rng = np.random.default_rng(
                random_state + fold_idx * 10_000 + int(round(p_val * 10_000))
            )
            rand_indices = _stratified_random_subset(y_train, n_subset=len(top_indices), rng=rng)

            X_train_rand, y_train_rand = X_train[rand_indices], y_train[rand_indices]
            best_model_rand = tune_hyperparameters(
                X_train_rand, y_train_rand, model, param_grid, inner_splits, random_state
            )
            y_pred_rand = best_model_rand.predict(X_test)
            brier_rand, ece_rand = _proba_metrics(best_model_rand, X_test, y_test)

            f1_rand = f1_score(y_test, y_pred_rand, zero_division=0)

            results.append({
                "Fold": fold_idx,
                "Model": model_name,
                "Method": "SEPM-RAND",
                "Top Percentage": p_val,
                "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred_rand),
                "F1 Score": f1_rand,
                "Accuracy Score": accuracy_score(y_test, y_pred_rand),
                "Brier Score": brier_rand,
                "ECE": ece_rand,
                "Shap Time": shap_time,
            })

            # Update best RAND per fold (tie -> smaller p)
            y_prob_rand = _get_probs(best_model_rand, X_test)
            if (f1_rand > best_rand_f1) or (np.isclose(f1_rand, best_rand_f1) and p_val < best_rand_p):
                best_rand_f1 = f1_rand
                best_rand_p = p_val
                best_rand_pred_record = {
                    "Fold": fold_idx,
                    "Model": model_name,
                    "Method": "SEPM-RAND",
                    "Top Percentage": p_val,
                    "y_true": y_test.astype(int),
                    "y_prob": y_prob_rand.astype(float),
                }

        # Append only best-p prediction records for this fold
        if best_sepm_pred_record is not None:
            best_fold_predictions.append(best_sepm_pred_record)
        if best_rand_pred_record is not None:
            best_fold_predictions.append(best_rand_pred_record)

        gc.collect()

    return pd.DataFrame(results), pd.DataFrame(best_fold_predictions)


def evaluate_models_using_nested_cv(
    X, y, model_configs,
    outer_splits, inner_splits, random_state,
    standardize, standardize_shap,
    top_percentages
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate multiple models using nested cross-validation with SEPM + SEPM-RAND.

    Returns
    -------
    best_scores : pd.DataFrame
        Aggregated table (best-p per fold then averaged), matching your reporting.
    all_results_df : pd.DataFrame
        Full per-fold metrics for all top_percentages.
    best_fold_predictions_df : pd.DataFrame
        Out-of-fold predictions for reliability diagrams:
          - TML per fold
          - SEPM best-p per fold
          - SEPM-RAND best-p per fold
    """
    all_results = []
    all_best_preds = []

    for model_config in model_configs:
        results_df, best_preds_df = nested_cross_validation(
            X, y, model_config,
            outer_splits, inner_splits, random_state,
            standardize, standardize_shap,
            top_percentages
        )
        all_results.append(results_df)
        all_best_preds.append(best_preds_df)

    all_results_df = pd.concat(all_results, ignore_index=True)
    best_fold_predictions_df = pd.concat(all_best_preds, ignore_index=True)

    best_scores = compute_max_metrics_for_each_method_model(all_results_df)

    return best_scores, all_results_df, best_fold_predictions_df


def compute_max_metrics_for_each_method_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize performance by selecting the best SEPM (and SEPM-RAND) configuration per outer fold,
    then averaging across folds (to match the single-model-per-fold structure of TML).

    Selection rule (per Model, Fold):
      - TML: keep the single row
      - SEPM: keep the row with the maximum F1 Score across Top Percentage values
              tie-breaker: choose the smaller Top Percentage
      - SEPM-RAND: same rule as SEPM

    Returns
    -------
    pd.DataFrame: mean ± SD across folds for each (Method, Model), plus Mean/SD selected Top Percentage.
    """
    required = {'Model', 'Method', 'Fold', 'F1 Score', 'Balanced Accuracy', 'Accuracy Score'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    def _pick_best(g: pd.DataFrame) -> pd.DataFrame:
        max_f1 = g['F1 Score'].max()
        cand = g[g['F1 Score'] == max_f1]
        if 'Top Percentage' in cand.columns:
            cand = cand.sort_values('Top Percentage', ascending=True)
        return cand.iloc[[0]]

    best_rows = []

    for (model_name, fold_idx), g in df.groupby(['Model', 'Fold']):
        tml = g[g['Method'] == 'TML']
        if not tml.empty:
            best_rows.append(tml.iloc[[0]])

        sepm = g[g['Method'] == 'SEPM']
        if not sepm.empty:
            best_rows.append(_pick_best(sepm))

        sepm_rand = g[g['Method'] == 'SEPM-RAND']
        if not sepm_rand.empty:
            best_rows.append(_pick_best(sepm_rand))

    best_df = pd.concat(best_rows, ignore_index=True)

    perf_cols = ['F1 Score', 'Balanced Accuracy', 'Accuracy Score']
    opt_perf = [c for c in ['Brier Score', 'ECE', 'Shap Time'] if c in best_df.columns]
    perf_cols_all = perf_cols + opt_perf

    n_folds = (
        best_df.groupby(['Method', 'Model'])['Fold']
        .nunique()
        .reset_index(name='N Folds')
    )

    agg_mean = best_df.groupby(['Method', 'Model'])[perf_cols_all].mean(numeric_only=True).reset_index()
    agg_std = best_df.groupby(['Method', 'Model'])[perf_cols_all].std(numeric_only=True).reset_index()

    agg_mean = agg_mean.rename(columns={c: f"Mean {c}" for c in perf_cols_all})
    agg_std = agg_std.rename(columns={c: f"SD {c}" for c in perf_cols_all})

    summary = agg_mean.merge(agg_std, on=['Method', 'Model'], how='left')
    summary = summary.merge(n_folds, on=['Method', 'Model'], how='left')

    if 'Top Percentage' in best_df.columns:
        p_stats = (
            best_df.groupby(['Method', 'Model'])['Top Percentage']
            .agg(**{'Mean Top Percentage': 'mean', 'SD Top Percentage': 'std'})
            .reset_index()
        )
        summary = summary.merge(p_stats, on=['Method', 'Model'], how='left')

    return summary


def run_pipeline_sepm_nested_cv(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    classifiers_dict: Dict[str, List[Dict[str, Any]]],
    top_percentages: List[float],
    standardize: bool = True,
    standardize_shap: bool = False,
    inner_splits: int = 5,
    outer_splits: int = 10,
    test_size: float = 0.3,
    random_state: int = 30,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Run both train-test split and nested cross-validation pipelines for multiple datasets.
    Now also returns best_fold_predictions for reliability diagrams.
    """

    def run_for_single_dataset(dataset_name, X, y, classifiers):
        if not classifiers:
            print(f"No classifiers specified for dataset {dataset_name}, skipping...")
            return dataset_name, None, None, None, None

        print(f"Running pipeline for {dataset_name} dataset...")

        train_test_results = train_test_split_evaluation(
            X, y, test_size, inner_splits, random_state,
            standardize, standardize_shap, classifiers, top_percentages
        )

        nested_cv_results, all_nested_cv_results, best_fold_predictions = evaluate_models_using_nested_cv(
            X, y, classifiers,
            outer_splits, inner_splits, random_state,
            standardize, standardize_shap,
            top_percentages
        )

        return dataset_name, train_test_results, nested_cv_results, all_nested_cv_results, best_fold_predictions

    datasets_results = Parallel(n_jobs=n_jobs)(
        delayed(run_for_single_dataset)(dataset_name, X, y, classifiers_dict.get(dataset_name))
        for dataset_name, (X, y) in datasets.items()
    )

    all_results = {
        dataset_name: {
            'train_test': train_results,
            'nested_cv': nested_results,
            'all_nested_cv_results': all_nested_results,
            'best_fold_predictions': best_fold_predictions
        }
        for dataset_name, train_results, nested_results, all_nested_results, best_fold_predictions
        in datasets_results
        if train_results is not None
    }

    gc.collect()
    return all_results
