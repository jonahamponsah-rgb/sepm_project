"""Reproducible SEPM simulation aligned with the manuscript design.

Follows the Simulation Study section of the SEPM manuscript:
- Synthetic data with ~70 % informative features, pure-noise remainder,
  heavy-tailed outliers, and two binary predictors.
- Class imbalance induced by calibrated intercept (equivalent to thresholding
  the continuous latent outcome).
- Evaluation on three complementary test groups:
    FTG  = full test set
    HSTG = high-SHAP test subset (observations with large SHAP values)
    LSTG = low-SHAP test subset  (observations with minimal SHAP contributions)
- Primary comparison: SEPM vs TML (SEPM-RAND retained as ablation control).

Run this file in the shga_ctgan environment. No external data files, SHAP
package, TMC, or earlier pilot modules are required. Console users may call
main().
"""
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
from importlib.metadata import version
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, brier_score_loss
from xgboost import XGBClassifier, DMatrix

# ---------------------------------------------------------------------------
# Editable settings
# ---------------------------------------------------------------------------
OUTPUT_ROOT = Path.cwd() / "sepm_simulation_runs"

# Expanded grid that covers the range shown in the manuscript heatmaps while
# remaining practical for repeated sampling.  Larger sizes (10k–50k) can be
# added when running on a cluster.
TRAIN_SIZES = [100, 200, 500, 1000, 5000]
TARGET_PREVALENCES = [.10, .20, .30, .35]  # population P(Y=1)

N_REPETITIONS = 100      # reduced from 50 for practical runtime; raise for final runs
TEST_SIZE = 5000          # large independent test set (more stable than 70/30)
INNER_FOLDS = 5
SEED = 30
RANDOM_REPEATS = 3        # random repeats for SEPM-RAND ablation
SEARCH_JOBS = 3
PROPORTIONS = [.40, .50, .55, .60, .65, .70, .75, .80, .85, .90, .95, .97, .98, .99, 1.00]

# Small prespecified grid for reasonable runtime; identical for all methods.
PARAM_GRID = dict(n_estimators=[50, 100, 200], max_depth=[3, 5, 7], learning_rate=[0.01, 0.1, 0.2])


RARE_FRACTION = .10                 # unobserved effect-modifying audit stratum
CONTAMINATION_FRACTION = .05        # heavy-tailed outliers
CALIBRATION_SIZE = 100000
SAVE_PREDICTIONS = True
METRICS = ["f1", "balanced_accuracy", "ece", "brier"]

# HSTG / LSTG are no longer fixed fractions.
# They are defined from the SEPM-selected training proportion (chosen['SEPM']),
# exactly as in the original manuscript simulation code (sim.py).


def seed_for(*parts):
    return int(np.random.SeedSequence([SEED, *parts]).generate_state(1)[0])


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = 1 / (1 + np.exp(-z[positive]))
    e = np.exp(z[~positive])
    result[~positive] = e / (1 + e)
    return result


def latent_population(n, rng):
    """Generate predictors and latent linear predictor matching the manuscript
    simulation design.

    Approximately 70 % of features are informative (direct linear association
    with the outcome); the remainder are pure noise. Eight continuous features
    are drawn from a standard Gaussian; two binary features emulate categorical
    variables. A continuous latent outcome is formed as a linear combination of
    the informative features. An unobserved rare subgroup (effect-modifying
    audit stratum, never supplied to the classifier) is also generated for
    stratified evaluation.
    """
    # 10 features total → 7 informative ≈ 70 %
    X = np.column_stack([rng.normal(size=(n, 8)), rng.binomial(1, .4, size=(n, 2))])
    rare = rng.random(n) < RARE_FRACTION
    # Informative: continuous indices 0–4 and both binary features (8, 9).
    # Noise: continuous indices 5–7 (never appear in the linear predictor).
    linear = (1.2 * X[:, 0] - X[:, 1] + .8 * X[:, 2] + .6 * X[:, 3]
              - .5 * X[:, 4] + .8 * (X[:, 8] - .4) - .7 * (X[:, 9] - .4))
    # Effect modification inside the rare subgroup (adds heterogeneity).
    linear[rare] += -2.4 * X[rare, 0] + 2 * X[rare, 1] + .5
    return X, linear, rare


def calibrate_intercept(target):
    """Calibrate the intercept so that the population prevalence equals `target`.

    This is numerically equivalent to thresholding the continuous latent outcome
    at different cut-points to induce the desired class imbalance. Calibration
    is performed on a large independent sample and does not depend on any
    training or test draw.
    """
    rng = np.random.default_rng(seed_for(900))
    _, linear, _ = latent_population(CALIBRATION_SIZE, rng)
    lo, hi = -30., 30.
    for _ in range(80):
        mid = (lo + hi) / 2
        if sigmoid(linear + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def generate(n, intercept, seed):
    """Draw a finite sample under the manuscript data-generating process.

    1. Form the continuous latent outcome from the informative features.
    2. Convert to a binary response via the calibrated intercept (class imbalance).
    3. Introduce outliers into a subset of continuous features by adding draws
       from a heavy-tailed t-distribution (measurement contamination applied
       after the outcome has been generated from the clean predictors).
    """
    rng = np.random.default_rng(seed)
    X, linear, rare = latent_population(n, rng)
    probability = sigmoid(intercept + linear)
    y = rng.binomial(1, probability).astype(int)
    contaminated = rng.random(n) < CONTAMINATION_FRACTION
    # Outliers affect three informative continuous predictors.
    X[contaminated, :3] += 3 * rng.standard_t(df=3, size=(contaminated.sum(), 3))
    return X, y, rare, contaminated, probability


def pipeline(estimator):
    return Pipeline([("scaler", StandardScaler()), ("model", clone(estimator))])


def expected_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        strategy: str = "uniform",
) -> float:
    """
    Compute binary expected calibration error (ECE).

    Supported strategies:
    - 'uniform': Equal-width probability bins.
    - 'quantile': Adaptive, equal-frequency bins (same number of samples per bin).
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Calibration inputs cannot be empty.")
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob contains non-finite values.")
    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("Probabilities must be in [0, 1].")

    n_observations = len(y_true)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=False)
    elif strategy == "quantile":
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0
        bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=False)
        n_bins = len(bin_edges) - 1
    else:
        raise ValueError("Strategy must be 'uniform' or 'quantile'.")

    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_counts = np.bincount(bin_ids, minlength=n_bins)
    nonzero_mask = bin_counts > 0

    bin_prob_sums = np.bincount(bin_ids, weights=y_prob, minlength=n_bins)
    bin_true_sums = np.bincount(bin_ids, weights=y_true, minlength=n_bins)

    mean_probabilities = bin_prob_sums[nonzero_mask] / bin_counts[nonzero_mask]
    observed_positive_rates = bin_true_sums[nonzero_mask] / bin_counts[nonzero_mask]
    bin_weights = bin_counts[nonzero_mask] / n_observations

    ece = np.sum(bin_weights * np.abs(observed_positive_rates - mean_probabilities))
    return float(ece)


def binary_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if not len(y):
        return dict.fromkeys(METRICS, np.nan)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid predicted probabilities")
    pred = (p >= .5).astype(int)
    ece = expected_calibration_error(y, p, n_bins=10, strategy="uniform")
    return dict(
        f1=float(f1_score(y, pred, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else np.nan,
        ece=ece,
        brier=float(brier_score_loss(y, p)),
    )


def shap_scores(fitted, X):
    matrix = DMatrix(fitted[:-1].transform(X))
    booster = fitted.named_steps['model'].get_booster()
    values = booster.predict(matrix, pred_contribs=True, approx_contribs=False, strict_shape=True)[:, 0, :]
    margins = booster.predict(matrix, output_margin=True, strict_shape=True).reshape(-1)
    if not np.allclose(values.sum(axis=1), margins, rtol=1e-4, atol=1e-4):
        raise ValueError("Native SHAP additivity failed")
    return np.abs(values[:, :-1]).mean(axis=1)


def ranked_subset(scores, y, p):
    selected = np.flatnonzero(scores >= np.percentile(scores, 100*(1-p)))
    for label in np.unique(y):
        if label not in y[selected]:
            candidates = np.flatnonzero(y==label)
            selected = np.append(selected, candidates[np.argmax(scores[candidates])])
    return selected.astype(int)


def random_subset(y, k, seed):
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=2)
    positive = int(np.clip(round(k*counts[1]/len(y)), max(1, k-counts[0]), min(counts[1], k-1)))
    return np.concatenate([rng.choice(np.flatnonzero(y==0), k-positive, replace=False),
                           rng.choice(np.flatnonzero(y==1), positive, replace=False)])


def select_proportions(X, y, estimator, cv_seed, sampling_seed, folder, n_inner_folds=None):
    # Use the same fold count as the hyperparameter search in this repetition.
    # Defaults to the prespecified value for backward compatibility.
    n_inner_folds = INNER_FOLDS if n_inner_folds is None else int(n_inner_folds)
    records, allocations = [], []
    splitter = StratifiedKFold(n_inner_folds, shuffle=True, random_state=cv_seed)
    for fold, (train, validation) in enumerate(splitter.split(X, y)):
        allocations.extend(dict(inner_fold=fold, training_row=int(i), role=role)
                           for role, indices in [('train',train),('validation',validation)] for i in indices)
        baseline = pipeline(estimator).fit(X[train], y[train])
        scores = shap_scores(baseline, X[train])
        for p in PROPORTIONS:
            selected = ranked_subset(scores, y[train], p)
            candidates = [('SEPM',0,selected)]
            candidates += [('SEPM-RAND',rep,random_subset(y[train],len(selected),
                             int(np.random.SeedSequence([sampling_seed,fold,rep]).generate_state(1)[0])))
                           for rep in range(RANDOM_REPEATS)]
            for method, rep, indices in candidates:
                fitted = pipeline(estimator).fit(X[train[indices]], y[train[indices]])
                prediction = fitted.predict_proba(X[validation])[:,1] >= .5
                records.append(dict(inner_fold=fold, method=method, random_repeat=rep,
                                    proportion=p, n_selected=len(indices),
                                    validation_f1=float(f1_score(y[validation],prediction,zero_division=0))))
    frame = pd.DataFrame(records)
    fm = frame.groupby(['method','proportion','inner_fold'],as_index=False).validation_f1.mean()
    means = fm.groupby(['method','proportion'],as_index=False).validation_f1.mean()
    chosen = {m:float(g.sort_values(['validation_f1','proportion'],ascending=[False,False]).iloc[0].proportion)
              for m,g in means.groupby('method')}
    frame.to_csv(folder/'inner_selection.csv',index=False)
    means.to_csv(folder/'selection_candidates.csv',index=False)
    pd.DataFrame(allocations).to_csv(folder/'inner_allocations.csv',index=False)
    return chosen


def summarize(records, retention, output):
    frame = pd.DataFrame(records)
    frame.to_csv(output/'repetition_results.csv',index=False)
    keys = ['scenario','n_train','target_prevalence','repetition','method','test_group']
    # Random draws are averaged WITHIN each simulated dataset, not counted as
    # independent simulation repetitions. Undefined subgroup values remain missing.
    fm = frame.groupby(keys,as_index=False)[METRICS].mean()
    fm.to_csv(output/'repetition_means.csv',index=False)
    group_keys = ['scenario','n_train','target_prevalence','method','test_group']
    summary = fm.groupby(group_keys)[METRICS].agg(['mean','std','count'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    for metric in METRICS:
        summary[metric+'_mcse'] = summary[metric+'_std']/np.sqrt(summary[metric+'_count'])
    summary.reset_index().to_csv(output/'summary.csv',index=False)
    contrasts = []
    pair_keys = ['scenario','n_train','target_prevalence','test_group']
    for key,g in fm.groupby(pair_keys):
        for comparator in ['TML','SEPM-RAND']:
            for metric in METRICS:
                pivot = g.pivot(index='repetition',columns='method',values=metric)
                if 'SEPM' not in pivot or comparator not in pivot:
                    continue
                differences = (pivot.SEPM-pivot[comparator]).dropna()
                n = len(differences)
                sd = float(differences.std(ddof=1)) if n>1 else np.nan
                contrasts.append(dict(zip(pair_keys,key),comparison='SEPM minus '+comparator,
                                      metric=metric,n_repetitions=n,mean_difference=differences.mean(),
                                      sd_difference=sd,mcse_difference=sd/np.sqrt(n) if n else np.nan,
                                      favorable_direction='positive' if metric in ['f1','balanced_accuracy'] else 'negative'))
    pd.DataFrame(contrasts).to_csv(output/'paired_differences.csv',index=False)
    pd.DataFrame(retention).to_csv(output/'training_retention.csv',index=False)


def run_repetition(n_train, target, intercept, scenario, repetition, folder):
    started = perf_counter()
    cv_seed = seed_for(scenario,repetition,2)
    train_seed, test_seed = seed_for(scenario,repetition,0),seed_for(scenario,repetition,1)
    X,y,rare,contaminated,_ = generate(n_train,intercept,train_seed)
    train_class_counts = np.bincount(y, minlength=2)
    minority_count = int(train_class_counts.min())
    if minority_count < 2:
        # Stratified CV with a positive sample in each fold is impossible.
        raise ValueError(f'Insufficient minority cases for at least 2 folds: {minority_count}; no redraw')
    # Retain the exact prespecified synthetic draw and seed. Use fewer folds only
    # where the observed minority count does not permit five-fold stratification.
    effective_inner_folds = min(INNER_FOLDS, minority_count)
    if effective_inner_folds < INNER_FOLDS:
        print(f'ADAPTIVE_CV scenario={scenario} rep={repetition}: '
              f'minority={minority_count}, folds={effective_inner_folds}/{INNER_FOLDS}', flush=True)
    model = XGBClassifier(random_state=42,objective='binary:logistic',n_jobs=1)
    search = GridSearchCV(pipeline(model),{'model__'+k:v for k,v in PARAM_GRID.items()},
                          scoring='f1',cv=StratifiedKFold(effective_inner_folds,shuffle=True,random_state=cv_seed),
                          n_jobs=SEARCH_JOBS,error_score='raise',refit=True)
    search.fit(X,y)
    fixed = clone(search.best_estimator_.named_steps['model'])
    chosen = select_proportions(X,y,fixed,cv_seed,seed_for(scenario,repetition,3),folder,
                                n_inner_folds=effective_inner_folds)
    scores = shap_scores(search.best_estimator_,X)
    indices = ranked_subset(scores,y,chosen['SEPM'])
    plans = [('TML',0,1.0,np.arange(n_train),search.best_estimator_),
             ('SEPM',0,chosen['SEPM'],indices,None)]
    # Same candidate count rule as the main SEPM-RAND pilot. The final chosen
    # proportions can differ, so final SEPM and RAND sizes need not match.
    k = len(ranked_subset(scores,y,chosen['SEPM-RAND']))
    for rep in range(RANDOM_REPEATS):
        ids = random_subset(y,k,seed_for(scenario,repetition,4,rep))
        plans.append(('SEPM-RAND',rep,chosen['SEPM-RAND'],ids,None))
    trained = []
    retention = []
    selected_rows = []
    for method,rep,p,ids,fitted in plans:
        if fitted is None:
            fitted = pipeline(fixed).fit(X[ids],y[ids])
        trained.append((method,rep,fitted))
        retention.append(dict(scenario=scenario,n_train=n_train,target_prevalence=target,
                              repetition=repetition,method=method,random_repeat=rep,
                              proportion=p,n_selected=len(ids),retained_fraction=len(ids)/n_train,
                              n_positive_selected=int(y[ids].sum()),n_rare_train=int(rare.sum()),
                              n_rare_selected=int(rare[ids].sum()),
                              rare_retention_rate=float(rare[ids].sum()/rare.sum()) if rare.sum() else np.nan,
                              common_retention_rate=float((~rare[ids]).sum()/(~rare).sum()) if (~rare).sum() else np.nan,
                              n_contaminated_train=int(contaminated.sum()),
                              n_contaminated_selected=int(contaminated[ids].sum())))
        selected_rows.extend(dict(method=method,random_repeat=rep,training_row=int(i)) for i in ids)
    # Test generation and prediction happen only AFTER every selection and fit.
    Xt, yt, rt, ct, truth = generate(TEST_SIZE, intercept, test_seed)

    # ------------------------------------------------------------------
    # HSTG / LSTG definition (matches original manuscript code in sim.py)
    # ------------------------------------------------------------------
    # 1. Compute mean |SHAP| on the test set using the full-data model.
    # 2. Use the same proportion that SEPM selected on training (chosen['SEPM'])
    #    as the percentile cut on the test-set SHAP distribution.
    # 3. high_shap = top fraction; low_shap = complement (everything below).
    test_shap_full = shap_scores(search.best_estimator_, Xt)
    sepm_proportion = chosen['SEPM']
    high_cut = np.percentile(test_shap_full, 100 * (1 - sepm_proportion))
    high_mask = test_shap_full >= high_cut
    low_mask = ~high_mask   # complementary set, as in the original code

    groups = {
        'full': np.ones(TEST_SIZE, dtype=bool),   # FTG
        'high_shap': high_mask,                   # HSTG
        'low_shap': low_mask,                     # LSTG
    }

    records, predictions = [], []
    for method, rep, fitted in trained:
        probability = fitted.predict_proba(Xt)[:, 1]

        for group, mask in groups.items():
            records.append(dict(
                scenario=scenario, n_train=n_train, target_prevalence=target,
                repetition=repetition, method=method, random_repeat=rep,
                test_group=group,
                n_test=int(mask.sum()), n_test_positive=int(yt[mask].sum()),
                **binary_metrics(yt[mask], probability[mask])
            ))

        if SAVE_PREDICTIONS:
            predictions.append(pd.DataFrame(dict(
                method=method, random_repeat=rep,
                test_row=np.arange(TEST_SIZE),
                y_true=yt, y_probability=probability,
                test_shap=test_shap_full,
                high_shap=high_mask, low_shap=low_mask,
                sepm_proportion_used=sepm_proportion,
                clean_outcome_probability=truth
            )))
    pd.DataFrame(dict(training_row=np.arange(n_train),y=y,rare_group=rare,contaminated=contaminated,
                      sepm_score=scores)).to_csv(folder/'training_audit.csv',index=False)
    pd.DataFrame(selected_rows).to_csv(folder/'selected_rows.csv',index=False)
    if SAVE_PREDICTIONS:
        pd.concat(predictions,ignore_index=True).to_csv(folder/'test_predictions.csv.gz',index=False,compression='gzip')
    # Persist effective CV folds in every repetition-level result and audit file.
    for row in records + retention:
        row['inner_folds_used'] = effective_inner_folds
    info = dict(train_seed=train_seed,test_seed=test_seed,cv_seed=cv_seed,intercept=intercept,
                train_class_counts=train_class_counts.tolist(),test_class_counts=np.bincount(yt,minlength=2).tolist(),
                requested_inner_folds=INNER_FOLDS, effective_inner_folds=effective_inner_folds,
                observed_minority_count=minority_count,
                adaptive_cv_applied=bool(effective_inner_folds < INNER_FOLDS),
                chosen_proportions=chosen,best_parameters=search.best_params_,seconds=perf_counter()-started)
    (folder/'repetition_metadata.json').write_text(json.dumps(info,indent=2),encoding='utf8')
    return records,retention,info


def main():
    if N_REPETITIONS<1 or TEST_SIZE<1 or INNER_FOLDS<2 or RANDOM_REPEATS<1:
        raise ValueError('Invalid repetition, sample-size, or fold setting')
    if any(p<=0 or p>1 for p in PROPORTIONS) or 1.0 not in PROPORTIONS:
        raise ValueError('Proportions must lie in (0,1] and include 1.0')
    if any(q<=0 or q>=1 for q in TARGET_PREVALENCES):
        raise ValueError('Target prevalence must lie in (0,1)')
    output = OUTPUT_ROOT/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    metadata = dict(
        status='running',
        created_utc=datetime.now(timezone.utc).isoformat(),
        train_sizes=TRAIN_SIZES,
        target_prevalences=TARGET_PREVALENCES,
        n_repetitions=N_REPETITIONS,
        test_size=TEST_SIZE,
        inner_folds=INNER_FOLDS,
        inner_fold_policy='min(requested folds, observed minority count), minimum 2; no redraw',
        seed=SEED,
        proportions=PROPORTIONS,
        param_grid=PARAM_GRID,
        random_repeats=RANDOM_REPEATS,
        rare_fraction=RARE_FRACTION,
        contamination_fraction=CONTAMINATION_FRACTION,
        calibration_size=CALIBRATION_SIZE,
        hstg_lstg_definition='uses SEPM-selected training proportion on test-set SHAP (manuscript original)',
        search_jobs=SEARCH_JOBS,
        model_jobs=1,
        versions={p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'xgboost']},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                      if '__file__' in globals() else None,
        completed_repetitions=0,
        experiment=('manuscript-aligned simulation: FTG/HSTG/LSTG evaluation; '
                    'independent large test set; training-only CV; SEPM vs TML + RAND ablation')
    )
    metadata_path=output/'run_metadata.json'
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf8')
    records,retention = [],[]
    started = perf_counter()
    try:
        intercepts={q:calibrate_intercept(q) for q in TARGET_PREVALENCES}
        metadata['intercepts']={str(k):v for k,v in intercepts.items()}
        for ni,n_train in enumerate(TRAIN_SIZES):
            for pi,target in enumerate(TARGET_PREVALENCES):
                # Stable scenario seeds if size/prevalence lists are later extended.
                scenario_seed=int(np.random.SeedSequence([n_train,int(round(target*1000000))]).generate_state(1)[0])
                scenario=f'n{n_train}_prev{target:g}'
                for repetition in range(N_REPETITIONS):
                    folder=output/scenario/f'repetition_{repetition:02d}'
                    folder.mkdir(parents=True)
                    # Integer scenario identity is used for RNG; readable label for outputs.
                    new,keep,info=run_repetition(n_train,target,intercepts[target],scenario_seed,repetition,folder)
                    for row in new+keep:
                        row['scenario']=scenario
                    records.extend(new);retention.extend(keep)
                    summarize(records,retention,output)
                    metadata['completed_repetitions']+=1
                    metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf8')
                    print(f'{scenario}: repetition {repetition+1}/{N_REPETITIONS}; '
                          f'p={info["chosen_proportions"]}; seconds={info["seconds"]:.1f}',flush=True)
        metadata['status']='complete'
    except Exception as exc:
        metadata.update(status='failed',error=repr(exc))
        raise
    finally:
        metadata['elapsed_seconds']=perf_counter()-started
        metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf8')
        print('Saved results:',output.resolve(),flush=True)


if __name__=='__main__':
    main()