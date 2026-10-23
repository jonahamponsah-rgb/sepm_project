# SEPM clinical analysis: diabetes. Scientific settings retained from the supplied final script.
from pathlib import Path
from time import perf_counter
from importlib.metadata import version
import json
import platform

import numpy as np
import pandas as pd
import shap

from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, balanced_accuracy_score, brier_score_loss
from xgboost import XGBClassifier, DMatrix

def ranked_subset(scores, y, proportion):
    """ SHAP-percentile selection rule."""
    scores = np.asarray(scores).reshape(-1)

    threshold = np.percentile(scores, 100 * (1 - proportion))
    selected = np.flatnonzero(scores >= threshold)

    # safeguard: add the highest-SHAP patient
    # from any missing outcome class.
    missing_classes = (
        set(np.unique(y)) - set(np.unique(y[selected]))
    )

    for label in missing_classes:
        candidates = np.flatnonzero(y == label)
        extra = candidates[np.argmax(scores[candidates])]
        selected = np.append(selected, extra)

    return selected.astype(int)

def random_subset(y, k, seed):
    """
    Random subset matching SEPM's actual selected count.
    Approximately preserves outcome proportions without requiring
    the unselected remainder to contain both classes.
    """
    y = np.asarray(y)
    n = len(y)

    if k >= n:
        return np.arange(n)

    classes, counts = np.unique(y, return_counts=True)

    if k < len(classes):
        raise ValueError("Subset must allow at least one per class.")

    rng = np.random.default_rng(seed)

    # Proportional allocation, with at least one patient per class.
    desired = k * counts / n
    allocated = np.floor(desired).astype(int)
    allocated = np.maximum(allocated, 1)
    allocated = np.minimum(allocated, counts)

    while allocated.sum() < k:
        available = np.flatnonzero(allocated < counts)
        chosen = available[
            np.argmax((desired - allocated)[available])
        ]
        allocated[chosen] += 1

    while allocated.sum() > k:
        removable = np.flatnonzero(allocated > 1)
        chosen = removable[
            np.argmax((allocated - desired)[removable])
        ]
        allocated[chosen] -= 1

    selected = np.concatenate([
        rng.choice(
            np.flatnonzero(y == label),
            size=int(amount),
            replace=False,
        )
        for label, amount in zip(classes, allocated)
    ])

    rng.shuffle(selected)
    return selected

def ece(y, probabilities, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.digitize(probabilities, edges[1:-1], right=True)
    error = 0.0

    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            error += mask.mean() * abs(
                probabilities[mask].mean() - y[mask].mean()
            )
    return float(error)

def evaluate(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predicted = (probabilities >= 0.5).astype(int)

    return {
        "f1": f1_score(y_test, predicted, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predicted
        ),
        "brier": brier_score_loss(y_test, probabilities),
        "ece": ece(y_test, probabilities),
    }, predicted, probabilities

DATA = Path(__file__).resolve().parents[1] / "data" / "pima-indians-diabetes.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "reruns" / "clinical" / "diabetes"

SEED = 30
OUTER_FOLDS = 10
INNER_FOLDS = 5
RANDOM_REPEATS = 3

# Parallelize grid-search fits; keep model-level threads limited.
SEARCH_JOBS = 3

PROPORTIONS = [
    0.40, 0.50, 0.55, 0.60, 0.65,
    0.70, 0.75, 0.80, 0.85, 0.90,
    0.95, 0.97, 0.98, 0.99, 1.00,
]

diabetes_classifiers = [
    {
        'name': 'SVM',
        'classifier': SVC(random_state=42, probability=True),
        'param_grid': {'C': [5, 50, 100], 'kernel': ['rbf', 'linear', "poly"], 'gamma':['auto', 'scale']}
        },
    {
        'name': 'Logistic Regression',
        'classifier': LogisticRegression(max_iter=10000, random_state=42, solver='liblinear'),
        'param_grid': {'C':  [1, 10, 100]}
    },

    {
        'name': 'XGBoost',
        'classifier': XGBClassifier(random_state=42, objective='binary:logistic'),
        'param_grid': {'n_estimators': [50, 100, 150], 'learning_rate': [0.2, 0.3], 'max_depth': [None, 4]}
    },
    {
        'name': 'GBM',
        'classifier': GradientBoostingClassifier(random_state=42),
        'param_grid': {'n_estimators': [50, 100, 150], 'learning_rate': [0.2, 0.3], 'max_depth': [1, 2]}
    },
    {
        'name': 'Random Forest',
        'classifier': RandomForestClassifier(random_state=42),
        'param_grid': {'n_estimators': [50, 100, 150], 'max_depth': [None, 4]}
    },
]
CONFIGS = {item["name"]: (item["classifier"], item["param_grid"])
           for item in diabetes_classifiers}
for _estimator, _grid in CONFIGS.values():
    if "n_jobs" in _estimator.get_params():
        _estimator.set_params(n_jobs=1)


def make_pipeline(estimator):
    return Pipeline([
        ("imputer", IterativeImputer(
            estimator=LinearRegression(), max_iter=500, random_state=SEED
        )),
        ("scaler", StandardScaler()),
        ("model", clone(estimator)),
    ])


def shap_scores(fitted_pipeline, X_train, model_name):
    """Compute mean absolute feature SHAP for training patients."""
    started = perf_counter()
    transformed = fitted_pipeline[:-1].transform(X_train)
    model = fitted_pipeline.named_steps["model"]

    if model_name == "XGBoost":
        booster = model.get_booster()
        matrix = DMatrix(transformed)

        contributions = booster.predict(
            matrix,
            pred_contribs=True,
            approx_contribs=False,
            strict_shape=True,
        )

        # Binary classification:
        # shape = (patients, 1 output group, features + bias).
        if contributions.shape != (
            len(transformed),
            1,
            transformed.shape[1] + 1,
        ):
            raise ValueError(
                f"Unexpected XGBoost contribution shape: "
                f"{contributions.shape}"
            )

        # Exclude the final bias column from patient scoring.
        values = contributions[:, 0, :-1]

        if not np.isfinite(values).all():
            raise ValueError("Nonfinite XGBoost SHAP contributions.")

        # Verify contributions reconstruct the raw model prediction.
        margins = booster.predict(matrix, output_margin=True)
        reconstructed = contributions[:, 0, :].sum(axis=1)

        if not np.allclose(
            reconstructed,
            np.asarray(margins).reshape(-1),
            rtol=1e-4,
            atol=1e-4,
        ):
            raise ValueError(
                "XGBoost SHAP contributions failed the additivity check."
            )

        return (
            np.abs(values).mean(axis=1),
            perf_counter() - started,
        )

    elif model_name == "SVM":
        # summarize training background
        # with five cluster centers.
        background = shap.kmeans(
            transformed, min(5, len(transformed))
        )
        explainer = shap.KernelExplainer(
            model.predict_proba, background
        )
        values = explainer.shap_values(transformed)



    elif model_name == "Logistic Regression":
        explainer = shap.LinearExplainer(model, transformed)
        values = explainer.shap_values(transformed)

    else:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)

    # Support both older list and newer array SHAP outputs.
    if isinstance(values, list):
        if len(values) != 2:
            raise ValueError(
                f"{model_name}: expected two class outputs."
            )
        values = values[1]

    values = np.asarray(values)
    if values.ndim == 3:
        if values.shape[-1] != 2:
            raise ValueError(
                f"{model_name}: unexpected SHAP shape {values.shape}"
            )
        values = values[:, :, 1]

    if values.shape != transformed.shape:
        raise ValueError(
            f"{model_name}: SHAP shape {values.shape}; "
            f"expected {transformed.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{model_name}: nonfinite SHAP values.")

    scores = np.abs(values).mean(axis=1)
    return scores, perf_counter() - started


def select_proportions(X, y, estimator, model_name, outer_fold):
    """Freeze baseline hyperparameters and select proportions."""
    cv = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=SEED + 1000 + outer_fold,
    )
    records = []
    shap_seconds = 0.0

    for inner_fold, (train, validation) in enumerate(cv.split(X, y)):
        X_train, y_train = X[train], y[train]
        X_validation, y_validation = X[validation], y[validation]

        baseline = make_pipeline(estimator).fit(X_train, y_train)
        scores, seconds = shap_scores(
            baseline, X_train, model_name
        )
        shap_seconds += seconds

        for proportion in PROPORTIONS:
            selected = ranked_subset(scores, y_train, proportion)

            model = make_pipeline(estimator).fit(
                X_train[selected], y_train[selected]
            )
            records.append({
                "model": model_name,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "method": "SEPM",
                "proportion": proportion,
                "repeat": 0,
                "n_selected": len(selected),
                "validation_f1": f1_score(
                    y_validation,
                    (model.predict_proba(X_validation)[:, 1] >= 0.5).astype(int),
                    zero_division=0,
                ),
            })

            for repeat in range(RANDOM_REPEATS):
                seed = (
                    SEED
                    + outer_fold * 10000
                    + inner_fold * 100
                    + repeat
                )
                indices = random_subset(
                    y_train, len(selected), seed
                )
                model_random = make_pipeline(estimator).fit(
                    X_train[indices], y_train[indices]
                )
                records.append({
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "method": "SEPM-RAND",
                    "proportion": proportion,
                    "repeat": repeat,
                    "n_selected": len(indices),
                    "validation_f1": f1_score(
                        y_validation,
                        (model_random.predict_proba(X_validation)[:, 1] >= 0.5).astype(int),
                        zero_division=0,
                    ),
                })

    selection = pd.DataFrame(records)
    means = (
        selection.groupby(
            ["method", "proportion"], as_index=False
        )["validation_f1"].mean()
    )

    chosen = {}
    for method, group in means.groupby("method"):
        best = group.sort_values(
            ["validation_f1", "proportion"],
            ascending=[False, False],
        ).iloc[0]
        chosen[method] = float(best["proportion"])

    return chosen, selection, shap_seconds


def save_outputs(results, predictions, selections, timings):
    """Checkpoint completed folds and summarize available results."""
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT / "fold_results.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        OUTPUT / "outer_predictions.csv", index=False
    )
    pd.concat(selections, ignore_index=True).to_csv(
        OUTPUT / "inner_selection.csv", index=False
    )
    pd.DataFrame(timings).to_csv(
        OUTPUT / "fold_timings.csv", index=False
    )

    metrics = ["f1", "balanced_accuracy", "brier", "ece"]

    # Repetitions are averaged within folds, not treated as
    # independent folds.
    fold_means = (
        results_df.groupby(
            ["model", "fold", "method"], as_index=False
        )[metrics].mean()
    )
    fold_means.to_csv(
        OUTPUT / "fold_means.csv", index=False
    )

    summary = (
        fold_means.groupby(["model", "method"])[metrics]
        .agg(["mean", "std"])
    )
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns
    ]
    summary.reset_index().to_csv(
        OUTPUT / "summary.csv", index=False
    )


def main():
    # Prevent accidental overwriting of a previous run.
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"{OUTPUT.resolve()} already contains files. "
            "Choose a new OUTPUT folder for this run."
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA, header=None)
    data.columns = ["Pregnancies", "Plasma", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Class"]
    # Match the original loader; zero pregnancies remain valid.
    missing_columns = ["Plasma", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    data[missing_columns] = data[missing_columns].replace(0, np.nan)
    if data["Class"].isna().any():
        raise ValueError("Missing outcome labels.")
    mapping = {0: 0, 1: 1}
    if set(data["Class"].unique()) != set(mapping):
        raise ValueError("Unexpected diabetes outcome labels.")
    X = data.drop(columns="Class").to_numpy(dtype=float)
    y = data["Class"].to_numpy(dtype=int)

    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected labels 0 and 1.")
    if np.min(np.bincount(y)) < OUTER_FOLDS:
        raise ValueError("Too few cases for the outer split.")

    print(
        f"Patients={len(y)}, predictors={X.shape[1]}, "
        f"class counts={np.bincount(y).tolist()}",
        flush=True,
    )

    versions = {
        name: version(name)
        for name in [
            "numpy", "pandas", "scipy",
            "scikit-learn", "shap", "xgboost",
        ]
    }
    metadata = {
        "status": "running",
        "dataset_name": "Pima Indians Diabetes",
        "missing_policy": "zeros in Plasma, BloodPressure, SkinThickness, Insulin and BMI become NaN; training-partition iterative imputation",
        "decision_threshold": 0.5,
        "inner_selection_prediction_rule": "positive-class probability >= 0.5",
        "baseline_gridsearch_prediction_rule": "classifier native predict via F1 scorer",
        "outer_test_prediction_rule": "positive-class probability >= 0.5",
        "selection_rule": "highest mean inner-validation F1; larger p on exact tie",
        "label_mapping": mapping,
        "missing_feature_entries": int(data.drop(columns="Class").isna().sum().sum()),
        "missing_by_feature": data.drop(columns="Class").isna().sum().astype(int).to_dict(),
        "dataset": str(DATA.resolve()),
        "n_patients": len(y),
        "n_predictors": X.shape[1],
        "seed": SEED,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "random_repeats": RANDOM_REPEATS,
        "proportions": PROPORTIONS,
        "tie_rule": "larger proportion on exact validation-F1 tie",
        "selection": "percentile threshold plus class-presence safeguard",
        "tuning": "baseline hyperparameters first, then fixed during proportion selection",
        "hyperparameter_grids": {
            name: grid for name, (_, grid) in CONFIGS.items()
        },
        "python": platform.python_version(),
        "versions": versions,
    }

    def write_metadata():
        (OUTPUT / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    write_metadata()
    started = perf_counter()
    results, predictions, selections, timings = [], [], [], []

    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=SEED,
    )
    # Same outer patients for every classifier.
    outer_splits = list(outer_cv.split(X, y))

    try:
        for model_number, (name, (base, grid)) in enumerate(
            CONFIGS.items()
        ):
            model_started = perf_counter()

            for fold, (train, test) in enumerate(outer_splits):
                # Seed approximate SHAP operations reproducibly.
                np.random.seed(SEED + model_number * 1000 + fold)
                fold_started = perf_counter()
                X_train, y_train = X[train], y[train]

                if np.min(np.bincount(y_train)) < INNER_FOLDS:
                    raise ValueError(
                        f"{name}, fold {fold}: too few minority cases."
                    )

                tune_started = perf_counter()
                search = GridSearchCV(
                    make_pipeline(base),
                    {
                        f"model__{key}": values
                        for key, values in grid.items()
                    },
                    scoring="f1",
                    cv=StratifiedKFold(
                        n_splits=INNER_FOLDS,
                        shuffle=True,
                        random_state=SEED + fold,
                    ),
                    n_jobs=SEARCH_JOBS,
                    error_score="raise",
                    refit=True,
                )
                search.fit(X_train, y_train)
                tuning_seconds = perf_counter() - tune_started

                baseline = search.best_estimator_
                fixed_estimator = clone(
                    baseline.named_steps["model"]
                )

                selection_started = perf_counter()
                chosen, selection, inner_shap_seconds = (
                    select_proportions(
                        X_train, y_train, fixed_estimator, name, fold
                    )
                )
                selection_seconds = (
                    perf_counter() - selection_started
                )
                selections.append(selection)

                scores, final_shap_seconds = shap_scores(
                    baseline, X_train, name
                )
                selected = ranked_subset(
                    scores, y_train, chosen["SEPM"]
                )

                models = [
                    ("TML", 0, 1.0, np.arange(len(train)), baseline),
                    (
                        "SEPM", 0, chosen["SEPM"], selected,
                        make_pipeline(fixed_estimator).fit(
                            X_train[selected], y_train[selected]
                        ),
                    ),
                ]

                random_k = len(ranked_subset(
                    scores, y_train, chosen["SEPM-RAND"]
                ))
                for repeat in range(RANDOM_REPEATS):
                    indices = random_subset(
                        y_train,
                        random_k,
                        SEED + 1_000_000 + fold * 100 + repeat,
                    )
                    models.append((
                        "SEPM-RAND",
                        repeat,
                        chosen["SEPM-RAND"],
                        indices,
                        make_pipeline(fixed_estimator).fit(
                            X_train[indices], y_train[indices]
                        ),
                    ))

                params = json.dumps(search.best_params_)
                for method, repeat, p, indices, fitted in models:
                    metrics, predicted, probabilities = evaluate(
                        fitted, X[test], y[test]
                    )
                    results.append({
                        "model": name,
                        "fold": fold,
                        "method": method,
                        "repeat": repeat,
                        "proportion": p,
                        "n_selected": len(indices),
                        "n_selected_negative": int(
                            np.sum(y_train[indices] == 0)
                        ),
                        "n_selected_positive": int(
                            np.sum(y_train[indices] == 1)
                        ),
                        "best_parameters": params,
                        **metrics,
                    })

                    for row, actual, pred, probability in zip(
                        test, y[test], predicted, probabilities
                    ):
                        predictions.append({
                            "model": name,
                            "fold": fold,
                            "method": method,
                            "repeat": repeat,
                            "patient_row": int(row),
                            "y_true": int(actual),
                            "y_pred": int(pred),
                            "y_probability": float(probability),
                        })

                seconds = perf_counter() - fold_started
                timings.append({
                    "model": name,
                    "fold": fold,
                    "baseline_tuning_seconds": tuning_seconds,
                    "proportion_selection_seconds": selection_seconds,
                    "inner_shap_seconds": inner_shap_seconds,
                    "final_shap_seconds": final_shap_seconds,
                    "fold_compute_seconds": seconds,
                })
                save_outputs(
                    results, predictions, selections, timings
                )

                print(
                    f"{name} | fold {fold + 1}/{OUTER_FOLDS} | "
                    f"proportions={chosen} | "
                    f"seconds={seconds:.1f}",
                    flush=True,
                )

            print(
                f"{name} completed in "
                f"{perf_counter() - model_started:.1f} seconds",
                flush=True,
            )

        metadata["status"] = "complete"

    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = repr(exc)
        raise

    finally:
        metadata["total_seconds"] = perf_counter() - started
        write_metadata()

    print("\nSaved results to:", OUTPUT.resolve())
    print(pd.read_csv(OUTPUT / "summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
