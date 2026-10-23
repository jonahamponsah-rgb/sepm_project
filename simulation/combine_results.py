#!/usr/bin/env python3
"""Combine and audit returned CHTC simulation archives without extracting them."""
import argparse
import io
import json
from pathlib import Path
import re
import tarfile
import pandas as pd
import numpy as np

METRICS = ['f1','balanced_accuracy','ece','brier']
KEYS = ['scenario','n_train','target_prevalence','repetition','method','test_group','random_repeat']
EXPECTED_SIZES = [100,200,500,1000,5000]
EXPECTED_PREVS = [.10,.20,.30,.35]


def member_bytes(tar, suffix):
    names = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f'Expected 1 {suffix}, got {len(names)}')
    return tar.extractfile(names[0]).read()


def maybe_frame(tar, suffix):
    matches = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(suffix)]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f'Multiple files ending in {suffix}')
    return pd.read_csv(io.BytesIO(tar.extractfile(matches[0]).read()))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--archives', type=Path, default=Path('.'))
    a.add_argument('--output', type=Path, default=Path('combined_simulation'))
    a.add_argument('--mode', choices=['smoke','full'], default='full')
    args = a.parse_args()
    paths = sorted(args.archives.glob(f'sepm_sim_*_{args.mode}.tar.gz'))
    if not paths:
        raise SystemExit('No matching archives found')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata, result_tables, ret_tables = [], [], []
    sources = set()
    job_ids = set()
    for path in paths:
        with tarfile.open(path, 'r:gz') as t:
            status = member_bytes(t, 'job_exit_status.txt').decode().strip()
            if status != '0':
                raise RuntimeError(f'{path.name}: job exited {status}')
            md = json.loads(member_bytes(t, '/batch_metadata.json'))
            if md['mode'] != args.mode or md['status'] not in ('complete','complete_with_invalid_draws'):
                raise RuntimeError(f'{path.name}: unexpected run status {md["status"]}')
            if md['job_tag'] in job_ids:
                raise ValueError(f'Duplicate job tag: {md["job_tag"]}')
            job_ids.add(md['job_tag'])
            sources.add(md['original_script_sha256'])
            metadata.append(md)
            rs = maybe_frame(t, '/repetition_results.csv')
            rt = maybe_frame(t, '/training_retention.csv')
            if rs is not None:
                result_tables.append(rs)
            if rt is not None:
                ret_tables.append(rt)
    if len(sources) != 1:
        raise ValueError('Archives do not all use the same original source script version')
    mdframe = pd.DataFrame([{**{k:m[k] for k in ('job_tag','mode','scenario','n_train','prevalence',
                       'rep_start','rep_count','status','elapsed_seconds','original_script_sha256')},
                            'n_complete':len(m['completed_repetitions']),
                            'n_invalid':len(m['invalid_repetitions'])} for m in metadata])
    mdframe.to_csv(args.output/'job_inventory.csv', index=False)
    completed_pairs, failed_pairs, intended_pairs = set(), set(), set()
    failures = []
    for m in metadata:
        pair=lambda r:(int(m['n_train']),int(round(m['prevalence']*100)),int(r))
        intended_pairs.update(pair(r) for r in m['expected_repetitions'])
        completed_pairs.update(pair(r) for r in m['completed_repetitions'])
        for invalid in m['invalid_repetitions']:
            failed_pairs.add(pair(invalid['repetition']))
            failures.append(dict(n_train=m['n_train'],prevalence=m['prevalence'],
                       repetition=invalid['repetition'],reason=invalid['error']))
    pd.DataFrame(failures,columns=['n_train','prevalence','repetition','reason']).to_csv(
        args.output/'invalid_repetitions.csv',index=False)
    adaptive_rows=[]
    for m in metadata:
        for rep in m.get('adaptive_cv_repetitions',[]):
            adaptive_rows.append(dict(job_tag=m['job_tag'], **rep))
    pd.DataFrame(adaptive_rows,columns=[
        'job_tag','repetition','n_train','prevalence','minority_count',
        'requested_folds','effective_folds']).to_csv(
            args.output/'adaptive_cv_repetitions.csv',index=False)
    if completed_pairs & failed_pairs:
        raise ValueError('Some repetition IDs are marked both successful and invalid')
    expected_pairset = {(n,p,r) for n in EXPECTED_SIZES for p in [10,20,30,35] for r in range(100)}
    missing_archives = sorted(expected_pairset - intended_pairs)
    missing_runs = sorted(intended_pairs - completed_pairs - failed_pairs)
    gaps = [dict(n_train=n, prevalence_percent=p, repetition=r,
                 reason=('no_job_archive' if (n,p,r) in missing_archives else 'not_completed_in_archive'))
            for n,p,r in sorted(set(missing_archives+missing_runs))]
    pd.DataFrame(gaps,columns=['n_train','prevalence_percent','repetition','reason']).to_csv(
        args.output/'missing_repetitions.csv',index=False)
    if result_tables:
        results=pd.concat(result_tables,ignore_index=True)
        retention=pd.concat(ret_tables,ignore_index=True)
        if results.duplicated(KEYS).any():
            raise ValueError('Duplicate output result rows across archives')
        if retention.duplicated(['scenario','repetition','method','random_repeat']).any():
            raise ValueError('Duplicate retention rows across archives')
        observed={ (int(row.n_train),int(round(row.target_prevalence*100)),int(row.repetition))
                   for row in results[['n_train','target_prevalence','repetition']].drop_duplicates().itertuples(index=False) }
        if observed!=completed_pairs:
            raise ValueError('Recorded result repetitions mismatch metadata')
        results.to_csv(args.output/'repetition_results.csv',index=False)
        retention.to_csv(args.output/'training_retention.csv',index=False)
        # Random repeats aggregated within each simulation replicate, not independent.
        keys=['scenario','n_train','target_prevalence','repetition','method','test_group']
        reps=results.groupby(keys,as_index=False)[METRICS].mean()
        reps.to_csv(args.output/'repetition_means.csv',index=False)
        grp=['scenario','n_train','target_prevalence','method','test_group']
        summary=reps.groupby(grp)[METRICS].agg(['mean','std','count'])
        summary.columns=['_'.join(c) for c in summary.columns]
        for metric in METRICS:
            summary[metric+'_mcse']=summary[metric+'_std']/np.sqrt(summary[metric+'_count'])
        summary.reset_index().to_csv(args.output/'summary.csv',index=False)
        pairs=['scenario','n_train','target_prevalence','test_group']
        contrast=[]
        for key,g in reps.groupby(pairs):
            for comparator in ['TML','SEPM-RAND']:
                pivot=g.pivot(index='repetition',columns='method',values=METRICS)
                for metric in METRICS:
                    if (metric,'SEPM') not in pivot.columns or (metric,comparator) not in pivot.columns:
                        continue
                    dif=(pivot[(metric,'SEPM')]-pivot[(metric,comparator)]).dropna()
                    n=len(dif)
                    contrast.append(dict(zip(pairs,key),comparison='SEPM minus '+comparator,
                        metric=metric,n_repetitions=n,mean_difference=float(dif.mean()) if n else np.nan,
                        sd_difference=float(dif.std(ddof=1)) if n>1 else np.nan,
                        mcse_difference=float(dif.sem()) if n>1 else np.nan,
                        favorable_direction='positive' if metric in ['f1','balanced_accuracy'] else 'negative'))
        pd.DataFrame(contrast).to_csv(args.output/'paired_differences.csv',index=False)
    report={'source_versions': sorted(sources),'mode':args.mode,'archives':len(paths),
            'intended_repetition_ids_in_archives':len(intended_pairs),
            'completed_repetitions':len(completed_pairs),
            'invalid_draws_without_redraw':len(failed_pairs),
            'missing_repetitions_no_archive_or_result':len(gaps),
            'expected_full_repetitions':len(expected_pairset)}
    (args.output/'audit_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if failed_pairs:
        print('WARNING: Invalid no-redraw samples are excluded; see invalid_repetitions.csv')
    if gaps:
        print('WARNING: Missing archives/repetitions; see missing_repetitions.csv')

if __name__=='__main__':
    main()
