"""Audit one completed CHTC result directory; no refitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def audit(run):
    meta = json.loads((run / 'run_metadata.json').read_text())
    if meta['status'] != 'complete':
        raise ValueError(f"Run not complete: {run} ({meta['status']})")
    if len(meta['datasets']) != 1:
        raise ValueError('Expected exactly one dataset per job')
    ds = meta['datasets'][0]
    directory = run / ds
    res = pd.read_csv(directory / 'fold_results.csv')
    preds = pd.read_csv(directory / 'outer_predictions.csv')
    expected = meta['tmc_permutations_per_seed'] * len(meta['tmc_seeds'])
    folds = sorted(res.fold.unique())
    if len(folds) != 1 or folds[0] != meta['outer_fold_index']:
        raise AssertionError('Wrong outer fold in results')
    for fold in folds:
        fdir = directory / f'fold_{fold:02d}'
        allocations = pd.read_csv(fdir / 'final_allocations.csv')
        assert allocations.patient_row.is_unique
        test = set(allocations.loc[allocations.role == 'test', 'patient_row'])
        valuation = set(allocations.loc[allocations.role == 'valuation', 'patient_row'])
        selected = pd.read_csv(fdir / 'selected_rows.csv')
        for method, group in selected.groupby('method'):
            assert not test.intersection(group.patient_row)
            if method != 'TML':
                assert not valuation.intersection(group.patient_row)
        inner = pd.read_csv(fdir / 'inner_allocations.csv')
        for _, group in inner.groupby('inner_fold'):
            assert group.patient_row.is_unique
            assert not test.intersection(group.patient_row)
        trace = pd.read_csv(fdir / 'final_tmc_trace.csv')
        assert np.allclose(trace.marginal_sum, trace.terminal_utility)
        assert len(trace) == expected, (len(trace), expected)
    for (fold, method, repeat), g in preds.groupby(['fold', 'method', 'repeat']):
        y = g.y_true.to_numpy(dtype=int)
        p = g.y_probability.to_numpy(dtype=float)
        pred = (p >= .5).astype(int)
        tp = np.sum((y == 1) & (pred == 1)); fp = np.sum((y == 0) & (pred == 1))
        fn = np.sum((y == 1) & (pred == 0)); tn = np.sum((y == 0) & (pred == 0))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
        ba = (tp / (tp + fn) + tn / (tn + fp)) / 2
        brier = np.mean((p - y) ** 2)
        ids = np.digitize(p, np.linspace(0, 1, 11)[1:-1], right=True)
        ece = sum(np.sum(ids == b) / len(y) * abs(y[ids == b].mean() - p[ids == b].mean())
                  for b in np.unique(ids))
        row = res[(res.fold == fold) & (res.method == method) & (res.repeat == repeat)].iloc[0]
        assert np.allclose([f1, ba, ece, brier],
                           row[['f1', 'balanced_accuracy', 'ece', 'brier']].to_numpy(float))
    print(f"PASS: {ds}, outer fold {folds[0]}, TMC permutations per stream {meta['tmc_permutations_per_seed']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', type=Path, default=Path('results'))
    args = ap.parse_args()
    runs = sorted(p.parent for p in args.results_dir.rglob('run_metadata.json'))
    if not runs:
        raise FileNotFoundError(f'No run_metadata.json under {args.results_dir}')
    for run in runs:
        audit(run)
    print(f'Audited {len(runs)} completed run(s)')


if __name__ == '__main__':
    main()
