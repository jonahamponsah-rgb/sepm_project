#!/usr/bin/env python3
"""Run one deterministic SEPM simulation scenario/repetition range on CHTC.

The original sepm_simulation.py is deliberately kept unchanged. All scientific
calculations and seeds are supplied by that original source.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np
import pandas as pd
import sepm_simulation as sim


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n-train', type=int, required=True)
    p.add_argument('--prevalence-percent', type=int, required=True)
    p.add_argument('--rep-start', type=int, required=True)
    p.add_argument('--rep-count', type=int, required=True)
    p.add_argument('--mode', choices=['smoke','full'], default='smoke')
    p.add_argument('--output-root', type=Path, default=Path('simulation_results'))
    p.add_argument('--search-jobs', type=int, default=2)
    p.add_argument('--omit-predictions', action='store_true')
    return p.parse_args()


def main():
    a = parse_args()
    target = a.prevalence_percent / 100.0
    if a.n_train not in sim.TRAIN_SIZES or target not in sim.TARGET_PREVALENCES:
        raise ValueError('Scenario is not among the 20 scenarios in the original script')
    if a.rep_start < 0 or a.rep_count <= 0 or a.rep_start + a.rep_count > sim.N_REPETITIONS:
        raise ValueError('Repetition range must fall within 0..99')
    if a.search_jobs < 1:
        raise ValueError('search-jobs must be >= 1')
    sim.SEARCH_JOBS = a.search_jobs
    sim.SAVE_PREDICTIONS = not a.omit_predictions
    if a.mode == 'smoke':
        # Deliberately reduced computational workload; not manuscript results.
        sim.INNER_FOLDS = 2
        sim.TEST_SIZE = 250
        sim.PROPORTIONS = [.80, 1.00]
        sim.RANDOM_REPEATS = 1
        sim.PARAM_GRID = dict(n_estimators=[10], max_depth=[2], learning_rate=[.1])
        sim.SAVE_PREDICTIONS = False
    scenario = f'n{a.n_train}_prev{target:g}'
    tag = f'n{a.n_train}_p{a.prevalence_percent}_r{a.rep_start}_c{a.rep_count}_{a.mode}'
    out = a.output_root / tag
    out.mkdir(parents=True, exist_ok=True)
    src = Path(sim.__file__).resolve()
    metadata = dict(status='running', job_tag=tag, mode=a.mode, scenario=scenario,
                    n_train=a.n_train, prevalence=target, rep_start=a.rep_start,
                    rep_count=a.rep_count, expected_repetitions=list(range(a.rep_start, a.rep_start+a.rep_count)),
                    completed_repetitions=[], invalid_repetitions=[], adaptive_cv_repetitions=[],
                    inner_fold_policy='min(requested folds, observed minority count), minimum 2; no redraw',
                    seed=sim.SEED, inner_folds=sim.INNER_FOLDS,
                    test_size=sim.TEST_SIZE, search_jobs=sim.SEARCH_JOBS,
                    save_predictions=sim.SAVE_PREDICTIONS, proportions=sim.PROPORTIONS,
                    param_grid=sim.PARAM_GRID, random_repeats=sim.RANDOM_REPEATS,
                    original_script_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                    created_utc=datetime.now(timezone.utc).isoformat())
    mpath = out / 'batch_metadata.json'
    mpath.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    started = perf_counter()
    records, retention = [], []
    try:
        intercept = sim.calibrate_intercept(target)
        metadata['calibrated_intercept'] = intercept
        scenario_seed = int(np.random.SeedSequence(
            [a.n_train, int(round(target*1000000))]).generate_state(1)[0])
        for rep in range(a.rep_start, a.rep_start+a.rep_count):
            folder = out / scenario / f'repetition_{rep:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            try:
                new, keep, info = sim.run_repetition(
                    a.n_train, target, intercept, scenario_seed, rep, folder)
            except ValueError as exc:
                # Do not redraw. With adaptive CV, only counts below two
                # are invalid because stratification is then impossible.
                if 'Insufficient minority cases' not in str(exc):
                    raise
                entry = dict(repetition=rep, error=str(exc),
                             policy='no redraw; fewer than two minority cases; excluded from completed results')
                metadata['invalid_repetitions'].append(entry)
                (folder / 'invalid_repetition.json').write_text(
                    json.dumps(entry, indent=2), encoding='utf-8')
                print(f'INVALID {scenario} rep={rep}: {exc}', flush=True)
                continue
            records.extend(new)
            retention.extend(keep)
            metadata['completed_repetitions'].append(rep)
            if info['adaptive_cv_applied']:
                metadata['adaptive_cv_repetitions'].append(dict(
                    repetition=rep, n_train=a.n_train, prevalence=target,
                    minority_count=info['observed_minority_count'],
                    requested_folds=info['requested_inner_folds'],
                    effective_folds=info['effective_inner_folds']))
            print(f'COMPLETE {scenario} repetition={rep} '
                  f'folds={info["effective_inner_folds"]} '
                  f'seconds={info["seconds"]:.1f}', flush=True)
            mpath.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        if records:
            sim.summarize(records, retention, out)
        metadata['status'] = ('complete_with_invalid_draws' if metadata['invalid_repetitions']
                              else 'complete')
    except Exception as exc:
        metadata['status'] = 'failed'
        metadata['error'] = repr(exc)
        metadata['traceback'] = traceback.format_exc()
        raise
    finally:
        metadata['elapsed_seconds'] = perf_counter()-started
        mpath.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(f'OUTPUT {out} STATUS {metadata["status"]}', flush=True)


if __name__ == '__main__':
    main()
