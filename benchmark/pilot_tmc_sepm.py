"""XGBoost comparison of SEPM and held-out-utility TMC sample selection.

Run main() in DataSpell; no argparse is used. Default settings are a runtime
pilot, not the manuscript experiment. Read TMC_SEPM_README.md before full runs.
"""
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import platform

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, brier_score_loss
from xgboost import XGBClassifier, DMatrix

# SETTINGS: change these here, including when using the DataSpell console.
DATA_DIR = Path("data")
OUTPUT_ROOT = Path.cwd() / "tmc_sepm_runs"
DATASETS = ["cervical"]  # full experiment: cervical, breast, diabetes, haberman
OUTER_FOLDS = 10
INNER_FOLDS = 5
OUTER_FOLD_LIMIT = 1  # Legacy: first N outer folds, set None for individual fold jobs.
OUTER_FOLD_INDEX = None  # Optional zero-based outer fold; selected from fixed 10-fold partition.
SEED = 30
SEARCH_JOBS = 3
VALUATION_FRACTION = 0.20
PROPORTIONS = [.40, .50, .55, .60, .65, .70, .75, .80, .85, .90,
               .95, .97, .98, .99, 1.00]
RANDOM_REPEATS = 3
TMC_SEEDS = [101]  # predeclare several independent streams for a full experiment.
TMC_PERMUTATIONS = 10  # per stream; pilot budget, NOT a convergence guarantee.
TMC_TOLERANCE = .01  # absolute F1 distance from the full-candidate utility.
TMC_PATIENCE = 5
TMC_MIN_FRACTION = .10
METRICS = ["f1", "balanced_accuracy", "ece", "brier"]
GRIDS = {
    "cervical": dict(n_estimators=[50, 100, 150], learning_rate=[.1, .3, 10], max_depth=[4, 6]),
    "breast": dict(n_estimators=[50, 100, 150], learning_rate=[.1, .3], max_depth=[1, 2]),
    "diabetes": dict(n_estimators=[50, 100, 150], learning_rate=[.1, .2, .3], max_depth=[None, 4]),
    "haberman": dict(n_estimators=[50, 100, 150], learning_rate=[.3, .5], max_depth=[2, 4]),
}


def load_data(name):
    files = {"cervical": "cervical_cancer.csv", "breast": "breast-cancer.data",
             "diabetes": "pima-indians-diabetes.csv", "haberman": "haberman.csv"}
    path = DATA_DIR / files[name]
    if name == "cervical":
        d = pd.read_csv(path)
        X, y = d.drop(columns="ca_cervix").to_numpy(float), d.ca_cervix.to_numpy(int)
    elif name == "breast":
        d = pd.read_csv(path, header=None, na_values=["?"])
        labels = d.iloc[:, 0]
        if labels.isna().any() or set(labels) != {"recurrence-events", "no-recurrence-events"}:
            raise ValueError("Unexpected breast outcome labels")
        X, y = d.iloc[:, 1:].to_numpy(), (labels == "recurrence-events").to_numpy(int)
    elif name == "diabetes":
        d = pd.read_csv(path, header=None)
        X, y = d.iloc[:, :-1].to_numpy(float), d.iloc[:, -1].to_numpy(int)
        for col in [1, 2, 3, 4, 5]:
            X[X[:, col] == 0, col] = np.nan
    else:
        d = pd.read_csv(path, header=None)
        if set(d.iloc[:, -1]) != {1, 2}:
            raise ValueError("Unexpected Haberman outcome labels")
        X, y = d.iloc[:, :-1].to_numpy(float), (d.iloc[:, -1] == 2).to_numpy(int)
    if set(np.unique(y)) != {0, 1} or np.bincount(y).min() < OUTER_FOLDS:
        raise ValueError("Binary labels and enough observations per class are required")
    return X, y, path


def pipeline(dataset, estimator):
    steps = []
    if dataset == "breast":
        steps += [("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                  ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]
    elif dataset == "diabetes":
        steps += [("imputer", IterativeImputer(estimator=LinearRegression(), max_iter=500,
                                               random_state=SEED, keep_empty_features=True))]
    return Pipeline(steps + [("scaler", StandardScaler()), ("model", clone(estimator))])


def metrics(y, p):
    pred = (np.asarray(p) >= .5).astype(int)
    ids = np.digitize(p, np.linspace(0, 1, 11)[1:-1], right=True)
    counts = np.bincount(ids, minlength=10)
    nonempty = counts > 0
    py = np.bincount(ids, weights=y, minlength=10)[nonempty] / counts[nonempty]
    pp = np.bincount(ids, weights=p, minlength=10)[nonempty] / counts[nonempty]
    return dict(f1=float(f1_score(y, pred, zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                ece=float(np.sum(counts[nonempty] * np.abs(py - pp)) / len(y)),
                brier=float(brier_score_loss(y, p)))


def percentile_subset(scores, y, proportion):
    """Original SEPM percentile rule, retaining cutoff ties and both classes."""
    selected = np.flatnonzero(scores >= np.percentile(scores, 100 * (1 - proportion)))
    for label in np.unique(y):
        if label not in y[selected]:
            candidates = np.flatnonzero(y == label)
            selected = np.append(selected, candidates[np.argmax(scores[candidates])])
    return np.asarray(selected, dtype=int)


def top_k(scores, y, k):
    """Exact count; deterministic score ties favor lower original row index."""
    if not 2 <= k <= len(y):
        raise ValueError("Invalid binary-class subset size")
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
    chosen = list(order[:k])
    for label in np.unique(y):
        if label not in y[chosen]:
            extra = next(i for i in order if y[i] == label)
            # Replace the lowest-ranked member; the remaining class still exists.
            chosen[-1] = extra
    assert len(set(chosen)) == k and len(np.unique(y[chosen])) == 2
    return np.asarray(chosen, dtype=int)


def random_k(y, k, seed):
    rng = np.random.default_rng(seed)
    # Proportional allocation with capacity limits and both classes present.
    counts = np.bincount(y, minlength=2)
    k1 = int(np.clip(round(k * counts[1] / len(y)), max(1, k - counts[0]), min(counts[1], k - 1)))
    return np.concatenate([rng.choice(np.flatnonzero(y == 0), k-k1, replace=False),
                           rng.choice(np.flatnonzero(y == 1), k1, replace=False)])


def split_pool(y, seed):
    pool, valuation = train_test_split(np.arange(len(y)), test_size=VALUATION_FRACTION,
                                      stratify=y, random_state=seed)
    assert not set(pool) & set(valuation)
    assert len(np.unique(y[pool])) == len(np.unique(y[valuation])) == 2
    return pool, valuation


def shap_scores(fitted, X):
    transformed = fitted[:-1].transform(X)
    booster = fitted.named_steps["model"].get_booster()
    matrix = DMatrix(transformed)
    contributions = booster.predict(matrix, pred_contribs=True, approx_contribs=False, strict_shape=True)
    contributions = contributions[:, 0, :]
    margins = booster.predict(matrix, output_margin=True, strict_shape=True).reshape(-1)
    if not np.allclose(contributions.sum(axis=1), margins, rtol=1e-4, atol=1e-4):
        raise ValueError("Native XGBoost SHAP additivity check failed")
    return np.abs(contributions[:, :-1]).mean(axis=1)


def estimate_tmc(X, y, X_value, y_value, dataset, estimator, seed, progress):
    """Fixed-budget truncated permutation estimator of an F1-utility game.

    Every coalition fit learns its own preprocessing. Empty coalition predicts
    class 0; one-class coalitions predict their sole class without fitting XGB.
    Remaining marginal increments are zero after truncation. Negative increments
    are retained. This implementation is not a call to the original authors' code.
    """
    def utility(indices):
        if len(indices) == 0:
            pred = np.zeros(len(y_value), dtype=int)
        elif len(np.unique(y[indices])) == 1:
            pred = np.full(len(y_value), y[indices[0]], dtype=int)
        else:
            fitted = pipeline(dataset, estimator).fit(X[indices], y[indices])
            pred = (fitted.predict_proba(X_value)[:, 1] >= .5).astype(int)
        return float(f1_score(y_value, pred, zero_division=0))

    n = len(y)
    full = utility(np.arange(n))
    rng = np.random.default_rng(seed)
    mean = np.zeros(n)
    m2 = np.zeros(n)
    records = []
    min_prefix = max(2, int(np.ceil(TMC_MIN_FRACTION * n)))
    for iteration in range(1, TMC_PERMUTATIONS + 1):
        started = perf_counter()
        order = rng.permutation(n)
        previous = 0.0  # empty-coalition F1 under class-zero convention
        consecutive = 0
        marginal = np.zeros(n)
        for j, index in enumerate(order):
            value = utility(order[:j+1])
            marginal[index] = value - previous
            previous = value
            consecutive = consecutive + 1 if abs(full - value) <= TMC_TOLERANCE else 0
            if j+1 >= min_prefix and consecutive >= TMC_PATIENCE:
                break
        delta = marginal - mean
        mean += delta / iteration
        m2 += delta * (marginal - mean)
        records.append(dict(permutation=iteration, evaluated_prefix=j+1, n_candidates=n,
                            truncated=(j+1 < n), full_utility=full, terminal_utility=previous,
                            marginal_sum=float(marginal.sum()), seconds=perf_counter()-started))
        if iteration == 1 or iteration % 5 == 0:
            print(f"{progress}: TMC {iteration}/{TMC_PERMUTATIONS}, prefix={j+1}/{n}", flush=True)
    # Monte Carlo SE only: it does not quantify truncation or sampling bias.
    se = np.sqrt(m2 / (TMC_PERMUTATIONS-1) / TMC_PERMUTATIONS) if TMC_PERMUTATIONS > 1 else np.full(n, np.nan)
    return mean, se, records


def rank_pool(X, y, Xv, yv, dataset, estimator, seed, context, pool_rows, output):
    times = {}
    started = perf_counter()
    baseline = pipeline(dataset, estimator).fit(X, y)
    fit_seconds = perf_counter()-started
    started = perf_counter()
    sepm = shap_scores(baseline, X)
    times["SEPM"] = fit_seconds + perf_counter()-started
    times["SEPM_baseline_fit"] = fit_seconds
    streams, logs, details = [], [], []
    started = perf_counter()
    for stream in TMC_SEEDS:
        values, se, trace = estimate_tmc(X, y, Xv, yv, dataset, estimator, seed+stream,
                                         f"{dataset}/{context}/seed{stream}")
        streams.append(values)
        logs.extend([dict(context=context, stream=stream, **r) for r in trace])
        details.extend(dict(context=context, stream=stream, patient_row=int(row),
                            tmc_value=float(v), mc_se=float(e)) for row, v, e in zip(pool_rows, values, se))
    times["TMC"] = perf_counter()-started
    # Equal fixed budgets across streams: their means can be averaged directly.
    tmc = np.mean(streams, axis=0)
    pd.DataFrame(logs).to_csv(output/f"{context}_tmc_trace.csv", index=False)
    pd.DataFrame(details).to_csv(output/f"{context}_tmc_stream_values.csv", index=False)
    pd.DataFrame(dict(patient_row=pool_rows, sepm_score=sepm, tmc_value=tmc)).to_csv(
        output/f"{context}_ranking.csv", index=False)
    return {"SEPM": sepm, "TMC": tmc}, times, baseline


def select_proportions(X, y, rows, dataset, estimator, fold, output):
    records, allocations = [], []
    costs = {m: 0.0 for m in ["SEPM", "TMC", "RAND"]}
    scoring = {m: 0.0 for m in ["SEPM", "TMC"]}
    cv = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED+fold)
    for inner, (train, validation) in enumerate(cv.split(X, y)):
        pool_local, value_local = split_pool(y[train], SEED+fold*100+inner)
        pool, valuation = train[pool_local], train[value_local]
        assert not (set(pool) & set(validation) or set(valuation) & set(validation))
        allocations.extend(dict(inner_fold=inner, patient_row=int(rows[i]), role=role)
                           for role, ids in [("candidate", pool), ("valuation", valuation), ("validation", validation)] for i in ids)
        scores, times, _ = rank_pool(X[pool], y[pool], X[valuation], y[valuation], dataset,
                                     estimator, SEED+fold*10000+inner*1000,
                                     f"inner_{inner}", rows[pool], output)
        for method in scoring:
            costs[method] += times[method]
            scoring[method] += times[method]
        for p in PROPORTIONS:
            selected_sepm = percentile_subset(scores["SEPM"], y[pool], p)
            k = min(len(pool), max(2, int(np.ceil(p * len(pool)))))
            candidates = [("SEPM", 0, selected_sepm),
                          ("TMC", 0, top_k(scores["TMC"], y[pool], k))]
            candidates += [("RAND", rep, random_k(y[pool], k, SEED+fold*10000+inner*100+rep))
                           for rep in range(RANDOM_REPEATS)]
            for method, rep, ids in candidates:
                started = perf_counter()
                fitted = pipeline(dataset, estimator).fit(X[pool[ids]], y[pool[ids]])
                pred = fitted.predict_proba(X[validation])[:, 1] >= .5
                f1 = float(f1_score(y[validation], pred, zero_division=0))
                costs[method] += perf_counter()-started
                records.append(dict(inner_fold=inner, method=method, repeat=rep, proportion=p,
                                    n_selected=len(ids), n_candidates=len(pool), validation_f1=f1))
    frame = pd.DataFrame(records)
    # Average random repetitions within folds before averaging folds.
    fold_means = frame.groupby(["method", "proportion", "inner_fold"]).validation_f1.mean().reset_index()
    means = fold_means.groupby(["method", "proportion"]).validation_f1.mean().reset_index()
    chosen = {method: float(g.sort_values(["validation_f1", "proportion"], ascending=[False, False]).iloc[0].proportion)
              for method, g in means.groupby("method")}
    frame.to_csv(output/"inner_selection.csv", index=False)
    means.to_csv(output/"selection_candidates.csv", index=False)
    pd.DataFrame(allocations).to_csv(output/"inner_allocations.csv", index=False)
    return chosen, costs, scoring


def run_dataset(dataset, root):
    X, y, path = load_data(dataset)
    output = root/dataset
    output.mkdir()
    base = XGBClassifier(random_state=42, objective="binary:logistic", n_jobs=1)
    results, predictions, runtime = [], [], []
    split_rows = []
    for fold, (train, test) in enumerate(StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED).split(X, y)):
        split_rows.extend(dict(fold=fold, patient_row=int(i), role=role) for role, ids in [("train", train), ("test", test)] for i in ids)
    pd.DataFrame(split_rows).to_csv(output/"outer_splits.csv", index=False)
    for fold, (train, test) in enumerate(StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED).split(X, y)):
        if OUTER_FOLD_INDEX is not None and fold != OUTER_FOLD_INDEX:
            continue
        if OUTER_FOLD_LIMIT is not None and fold >= OUTER_FOLD_LIMIT:
            break
        fold_output = output/f"fold_{fold:02d}"
        fold_output.mkdir()
        started = perf_counter()
        search = GridSearchCV(pipeline(dataset, base), {"model__"+k:v for k,v in GRIDS[dataset].items()},
                              scoring="f1", cv=StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED+fold),
                              n_jobs=SEARCH_JOBS, error_score="raise", refit=True)
        search.fit(X[train], y[train])
        tuning = perf_counter()-started
        estimator = clone(search.best_estimator_.named_steps["model"])
        (fold_output/"best_parameters.json").write_text(json.dumps(search.best_params_, indent=2), encoding="utf8")
        chosen, costs, scoring = select_proportions(X[train], y[train], train, dataset, estimator, fold, fold_output)
        a, b = split_pool(y[train], SEED+100000+fold)
        pool, valuation = train[a], train[b]
        assert not (set(pool) & set(test) or set(valuation) & set(test))
        pd.DataFrame([dict(patient_row=int(i), role=role) for role, ids in [("candidate",pool), ("valuation",valuation), ("test",test)] for i in ids]).to_csv(fold_output/"final_allocations.csv", index=False)
        scores, times, pool_model = rank_pool(X[pool], y[pool], X[valuation], y[valuation], dataset,
                                             estimator, SEED+1000000+fold*1000,
                                             "final", pool, fold_output)
        selected = percentile_subset(scores["SEPM"], y[pool], chosen["SEPM"])
        matched_k = len(selected)
        final_sets = [("TML", 0, 1.0, train, search.best_estimator_, "baseline"),
                      ("TML-POOL", 0, 1.0, pool, pool_model, "baseline"),
                      ("SEPM", 0, chosen["SEPM"], pool[selected], None, "optimized")]
        for method in ["TMC", "RAND"]:
            k = min(len(pool), max(2, int(np.ceil(chosen[method] * len(pool)))))
            for rep in range(RANDOM_REPEATS if method == "RAND" else 1):
                ids = top_k(scores["TMC"], y[pool], k) if method == "TMC" else random_k(y[pool], k, SEED+2000000+fold*100+rep)
                final_sets.append((method, rep, chosen[method], pool[ids], None, "optimized"))
                ids_match = top_k(scores["TMC"], y[pool], matched_k) if method == "TMC" else random_k(y[pool], matched_k, SEED+3000000+fold*100+rep)
                final_sets.append((method+"-MATCHED", rep, chosen["SEPM"], pool[ids_match], None, "matched"))
        final_seconds = {k:0.0 for k in ["SEPM","TMC","RAND","TMC-MATCHED","RAND-MATCHED","TML","TML-POOL"]}
        selected_rows = []
        for method, rep, p, indices, fitted, comparison in final_sets:
            started = perf_counter()
            if fitted is None:
                fitted = pipeline(dataset, estimator).fit(X[indices], y[indices])
            probabilities = fitted.predict_proba(X[test])[:, 1]
            values = metrics(y[test], probabilities)
            final_seconds[method] += perf_counter()-started
            results.append(dict(dataset=dataset, model="XGBoost", fold=fold, method=method, repeat=rep,
                                comparison=comparison, proportion=p, n_train=len(train), n_candidates=len(pool),
                                n_selected=len(indices), n_positive_selected=int(y[indices].sum()),
                                decision_threshold=.5, **values))
            selected_rows.extend(dict(method=method, repeat=rep, patient_row=int(i)) for i in indices)
            predictions.extend(dict(dataset=dataset, fold=fold, method=method, repeat=rep,
                                     patient_row=int(i), y_true=int(actual), y_probability=float(prob))
                               for i, actual, prob in zip(test, y[test], probabilities))
        pd.DataFrame(selected_rows).to_csv(fold_output/"selected_rows.csv", index=False)
        for method in ["SEPM","TMC","RAND"]:
            ranking = times[method] if method != "RAND" else 0.0
            selection_cost = costs[method]
            runtime.append(dict(dataset=dataset, fold=fold, method=method,
                                shared_baseline_tuning_seconds=tuning,
                                inner_scoring_seconds=scoring.get(method,0),
                                final_scoring_seconds=times.get(method,0),
                                selection_with_dependencies_seconds=selection_cost,
                                final_scoring_with_dependencies_seconds=ranking,
                                final_fit_predict_metric_seconds=final_seconds[method],
                                total_pipeline_seconds=tuning+selection_cost+ranking+final_seconds[method],
                                matched_extra_fit_predict_metric_seconds=final_seconds.get(method+"-MATCHED",0)))
        for method, fit_cost in [("TML", 0.0), ("TML-POOL", times["SEPM_baseline_fit"])]:
            runtime.append(dict(dataset=dataset, fold=fold, method=method,
                                shared_baseline_tuning_seconds=tuning,
                                inner_scoring_seconds=0.0, final_scoring_seconds=0.0,
                                selection_with_dependencies_seconds=0.0,
                                final_scoring_with_dependencies_seconds=fit_cost,
                                final_fit_predict_metric_seconds=final_seconds[method],
                                total_pipeline_seconds=tuning+fit_cost+final_seconds[method],
                                matched_extra_fit_predict_metric_seconds=0.0))
        pd.DataFrame(results).to_csv(output/"fold_results.csv", index=False)
        pd.DataFrame(predictions).to_csv(output/"outer_predictions.csv", index=False)
        pd.DataFrame(runtime).to_csv(output/"fold_timings.csv", index=False)
        fm = pd.DataFrame(results).groupby(["dataset","model","fold","method"], as_index=False)[METRICS].mean()
        fm.to_csv(output/"fold_means.csv", index=False)
        summary = fm.groupby(["dataset","model","method"])[METRICS].agg(["mean","std","count"])
        summary.columns = ["_".join(c) for c in summary.columns]
        summary.reset_index().to_csv(output/"summary.csv", index=False)
        print(f"{dataset}: completed outer fold {fold+1}/{OUTER_FOLDS}; choices={chosen}", flush=True)
    return dict(dataset=dataset, file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                n=len(y), predictors=X.shape[1], class_counts=np.bincount(y).tolist())


def main():
    if not DATASETS or any(d not in GRIDS for d in DATASETS):
        raise ValueError("Choose valid datasets")
    if TMC_PERMUTATIONS < 1 or not TMC_SEEDS or len(set(TMC_SEEDS)) != len(TMC_SEEDS):
        raise ValueError("TMC requires positive permutations and unique seeds")
    if OUTER_FOLD_LIMIT is not None and not 1 <= OUTER_FOLD_LIMIT <= OUTER_FOLDS:
        raise ValueError("Invalid outer-fold limit")
    if OUTER_FOLD_INDEX is not None and not 0 <= OUTER_FOLD_INDEX < OUTER_FOLDS:
        raise ValueError("Invalid outer-fold index")
    run = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True, exist_ok=False)
    metadata = dict(status="running", created_utc=datetime.now(timezone.utc).isoformat(),
                    datasets=DATASETS, outer_folds=OUTER_FOLDS, processed_fold_limit=OUTER_FOLD_LIMIT, outer_fold_index=OUTER_FOLD_INDEX,
                    inner_folds=INNER_FOLDS, seed=SEED, valuation_fraction=VALUATION_FRACTION,
                    proportions=PROPORTIONS, threshold=.5, tmc_utility="F1", tmc_seeds=TMC_SEEDS,
                    tmc_permutations_per_seed=TMC_PERMUTATIONS, tmc_tolerance=TMC_TOLERANCE,
                    tmc_patience=TMC_PATIENCE, tmc_min_fraction=TMC_MIN_FRACTION,
                    random_repeats=RANDOM_REPEATS, search_jobs=SEARCH_JOBS, estimator_jobs=1,
                    grids=GRIDS, platform=platform.platform(),
                    versions={p:version(p) for p in ["numpy","pandas","scikit-learn","xgboost"]},
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if "__file__" in globals() else None,
                    experiment="common candidate pool; held-out TMC valuation; separate from original SEPM runs",
                    pilot=(OUTER_FOLD_LIMIT is not None or OUTER_FOLD_INDEX is not None or TMC_PERMUTATIONS < 100 or len(TMC_SEEDS)<3),
                    inputs=[])
    start = perf_counter()
    (run/"run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    try:
        for dataset in DATASETS:
            metadata["inputs"].append(run_dataset(dataset, run))
        metadata["status"] = "complete"
    except Exception as exc:
        metadata.update(status="failed", error=repr(exc))
        raise
    finally:
        metadata["elapsed_seconds"] = perf_counter()-start
        (run/"run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
        print("Results:", run.resolve(), flush=True)


if __name__ == "__main__":
    main()
