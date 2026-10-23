"""Combine separately completed outer-fold archives after extraction.

Run locally after extracting sepm_tmc_*.tar.gz into a single folder.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ['f1', 'balanced_accuracy', 'ece', 'brier']


def combine(results_root, output_dir):
    metadata_paths = sorted(results_root.rglob('run_metadata.json'))
    if not metadata_paths:
        raise FileNotFoundError(f'No result metadata under {results_root}')
    records, predictions, times, run_configs = [], [], [], []
    seen = set()
    spec = None
    for mp in metadata_paths:
        run = mp.parent
        meta = json.loads(mp.read_text())
        if meta.get('status') != 'complete':
            raise ValueError(f'Unfinished job {run}: {meta.get("status")}')
        dataset = meta['datasets'][0]
        fold = meta['outer_fold_index']
        key = dataset, fold
        if key in seen:
            raise ValueError(f'Duplicate dataset/outer fold {key}: do not combine pilot and benchmark runs')
        seen.add(key)
        protocol = {k: meta[k] for k in ('seed', 'outer_folds', 'inner_folds', 'valuation_fraction',
                                         'proportions', 'tmc_seeds', 'tmc_permutations_per_seed',
                                         'random_repeats', 'threshold', 'grids')}
        if spec is None:
            spec = protocol
        elif spec != protocol:
            raise ValueError(f'Protocol mismatch in {run}; do not mix runs with different TMC budgets/grids')
        d = run / dataset
        r = pd.read_csv(d / 'fold_results.csv')
        if sorted(r.fold.unique()) != [fold] or r.dataset.nunique() != 1:
            raise AssertionError(f'Unexpected folds/datasets in {d}')
        records.append(r)
        predictions.append(pd.read_csv(d / 'outer_predictions.csv'))
        times.append(pd.read_csv(d / 'fold_timings.csv'))
        run_configs.append(dict(dataset=dataset, fold=fold, directory=str(run)))

    output_dir.mkdir(parents=True, exist_ok=True)
    r = pd.concat(records, ignore_index=True)
    r.to_csv(output_dir / 'fold_results_all.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output_dir / 'outer_predictions_all.csv', index=False)
    pd.concat(times, ignore_index=True).to_csv(output_dir / 'fold_timings_all.csv', index=False)
    # Within each fold, average random repeats before calculating cross-fold means/SD.
    fm = r.groupby(['dataset', 'model', 'fold', 'method'], as_index=False)[METRICS].mean()
    fm.to_csv(output_dir / 'fold_means_all.csv', index=False)
    summ = fm.groupby(['dataset', 'model', 'method'])[METRICS].agg(['mean', 'std', 'count'])
    summ.columns = ['_'.join(name) for name in summ.columns]
    summ.reset_index().to_csv(output_dir / 'summary_all_folds.csv', index=False)
    (output_dir / 'combined_metadata.json').write_text(json.dumps({
        'folds': run_configs, 'protocol': spec, 'distinct_folds': len(seen),
        'warning': 'Cross-fold SD is missing for methods with only one completed fold.'
    }, indent=2))
    print('Combined dataset/fold pairs:', sorted(seen))
    print('Wrote:', output_dir.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', type=Path, default=Path('results'))
    ap.add_argument('--output-dir', type=Path, default=Path('combined_results'))
    args = ap.parse_args()
    combine(args.results_dir, args.output_dir)


if __name__ == '__main__':
    main()
