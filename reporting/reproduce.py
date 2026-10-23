"""Verify saved records and reproduce the final tables and statistical results."""
from pathlib import Path
import hashlib,io,json,tarfile
import numpy as np
import pandas as pd
from .metrics import binary_metrics,METRICS
from .statistics import exact_signed_rank,friedman,holm,ranks

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduced'
DATASETS={'cervical':('Cervical cancer',72),'breast':('Breast recurrence',286),
          'diabetes':('Pima diabetes',768),'haberman':('Haberman',306)}
MODELS=['Logistic Regression','SVM','XGBoost','GBM','Random Forest']
METHODS=['TML','SEPM-RAND','SEPM']

def save(frame,name):
    frame.to_csv(OUT/name,index=False)

def compare(a,b,columns):
    assert np.allclose(a[columns].to_numpy(float),b[columns].to_numpy(float),atol=1e-10,rtol=1e-8,equal_nan=True)

def main_clinical():
    folds=[];counts=[];dataset_rows=[];checked=0
    for ds,(label,n) in DATASETS.items():
        folder=ROOT/'results/clinical'/ds
        meta=json.loads((folder/'run_metadata.json').read_text())
        assert meta['status']=='complete' and meta['n_patients']==n
        res=pd.read_csv(folder/'fold_results.csv');pred=pd.read_csv(folder/'outer_predictions.csv')
        sel=pd.read_csv(folder/'inner_selection.csv');saved=pd.read_csv(folder/'summary.csv')
        assert len(res)==250 and set(res.method)==set(METHODS)
        assert not pred.duplicated(['model','fold','method','repeat','patient_row']).any()
        assert np.array_equal(pred.y_pred,(pred.y_probability>=.5).astype(int))
        assert (pred.groupby('patient_row').y_true.nunique()==1).all()
        assert (pred.groupby('patient_row').fold.nunique()==1).all()
        assert pred.patient_row.nunique()==n
        fold_n=pred.groupby('fold').patient_row.nunique()
        assert set(fold_n.index)==set(range(10))
        assert (res.n_selected==res.n_selected_positive+res.n_selected_negative).all()
        assert (res.n_selected<=res.fold.map(lambda f:n-fold_n.loc[f])).all()
        for _,g in pred.groupby(['model','method','repeat']):assert g.patient_row.nunique()==len(g)==n
        for (model,fold,method,repeat),g in pred.groupby(['model','fold','method','repeat']):
            rr=res[(res.model==model)&(res.fold==fold)&(res.method==method)&(res['repeat']==repeat)].iloc[0]
            got=binary_metrics(g.y_true,g.y_probability)
            assert np.allclose([got[m] for m in METRICS],rr[METRICS].to_numpy(float),atol=1e-10)
            checked+=4
        fm=res.groupby(['model','fold','method'],as_index=False)[METRICS].mean()
        su=fm.groupby(['model','method'])[METRICS].agg(['mean','std'])
        su.columns=['_'.join(c) for c in su.columns]
        for _,rr in saved.iterrows():assert np.allclose(rr[su.columns].to_numpy(float),su.loc[(rr.model,rr.method)].to_numpy(float),atol=1e-10)
        means=sel.groupby(['model','outer_fold','method','proportion','inner_fold'],as_index=False).validation_f1.mean()
        means=means.groupby(['model','outer_fold','method','proportion'],as_index=False).validation_f1.mean()
        for (model,fold,method),g in means.groupby(['model','outer_fold','method']):
            best=g.sort_values(['validation_f1','proportion'],ascending=[False,False]).iloc[0].proportion
            got=res[(res.model==model)&(res.fold==fold)&(res.method==method)].proportion
            assert len(got) and np.allclose(got,best)
        c=res.groupby(['model','fold','method'],as_index=False)[['n_selected','n_selected_positive','n_selected_negative']].mean()
        c['n_train']=c.fold.map(lambda f:n-fold_n.loc[f]);c.insert(0,'dataset',label);counts.append(c)
        fm.insert(0,'dataset',label);folds.append(fm)
        y=pred.drop_duplicates('patient_row').y_true
        dataset_rows.append(dict(Dataset=label,n=n,Predictors=meta['n_predictors'],Positive=int(y.sum()),Negative=int((y==0).sum()),Missing_predictor_entries=meta['missing_feature_entries']))
    folds=pd.concat(folds,ignore_index=True);counts=pd.concat(counts,ignore_index=True)
    summary=folds.groupby(['dataset','model','method'])[METRICS].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
    retained=counts.groupby(['dataset','model','method'])[['n_selected','n_train']].sum()
    retained['retained_fraction']=retained.n_selected/retained.n_train
    summary=summary.merge(retained[['retained_fraction']].reset_index(),on=['dataset','model','method'])
    scores=summary.pivot(index=['dataset','model'],columns='method',values='f1_mean')[['TML','SEPM','SEPM-RAND']].sort_index()
    assert scores.shape==(20,3)
    omnibus,rr=friedman(scores.to_numpy());pairs=[]
    for first,second in [('SEPM','TML'),('SEPM','SEPM-RAND'),('TML','SEPM-RAND')]:
        diff=scores[first]-scores[second]
        pairs.append(dict(contrast=first+' vs '+second,mean_f1_difference=diff.mean(),median_f1_difference=diff.median(),
                          wins=int((diff>1e-10).sum()),ties=int((abs(diff)<=1e-10).sum()),losses=int((diff< -1e-10).sum()),**exact_signed_rank(diff)))
    pairs=pd.DataFrame(pairs);pairs['p_holm_three_comparisons']=holm(pairs.p_exact)
    avg=pd.DataFrame(dict(method=scores.columns,average_rank_1_best=np.mean([ranks(-r) for r in scores.to_numpy()],axis=0)))
    save(summary,'clinical_summary.csv');save(folds,'clinical_fold_means.csv');save(counts,'clinical_selection_counts.csv')
    save(scores.reset_index(),'clinical_task_f1.csv');save(pd.DataFrame([omnibus]),'clinical_friedman.csv');save(pairs,'clinical_wilcoxon_holm.csv');save(avg,'clinical_average_ranks.csv')
    save(pd.DataFrame(dataset_rows),'Table1_datasets.csv')
    rows=[];lookup=summary.set_index(['dataset','model','method'])
    for label,_ in DATASETS.values():
        for model in MODELS:
            row=dict(Dataset=label,Model=model)
            for method in METHODS:
                a=lookup.loc[(label,model,method)];row[method]=f'{a.f1_mean:.3f} ± {a.f1_std:.3f}'
            row['Retained']=f'{100*lookup.loc[(label,model,"SEPM"),"retained_fraction"]:.1f}%';rows.append(row)
    save(pd.DataFrame(rows),'Table2_clinical_F1.csv')
    order=['SEPM','SEPM-RAND','TML'];ranks_df=avg.set_index('method')
    combined=pairs.copy();combined.insert(0,'Method',order);combined.insert(1,'Average rank',[ranks_df.loc[m,'average_rank_1_best'] for m in order]);save(combined,'Table3_main_statistics.csv')
    save(summary,'Supplementary_clinical_four_metrics.csv')
    assert np.isclose(omnibus['friedman_chi_square'],16.126582278481,atol=1e-6)
    assert list(pairs.wins)==[18,16,7] and list(pairs.losses)==[2,4,12]
    return dict(prediction_metric_cells=checked,tasks=len(scores),comparisons=3),summary

def benchmark():
    results=[];timings=[];metas=[];checked=0;keys=set()
    source_hash=hashlib.sha256((ROOT/'benchmark/pilot_tmc_sepm.py').read_bytes()).hexdigest()
    archives=sorted((ROOT/'results/benchmark/fold_archives').glob('*.tar.gz'));assert len(archives)==40
    for path in archives:
        with tarfile.open(path) as t:files={m.name:t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
        def find(suffix):
            found=[v for k,v in files.items() if k.endswith(suffix)];assert len(found)==1,(path.name,suffix);return found[0]
        def frame(suffix):return pd.read_csv(io.BytesIO(find(suffix)))
        meta=json.loads(find('/run_metadata.json'));metas.append(meta)
        assert meta['status']=='complete' and find('job_exit_status.txt').strip()==b'0'
        assert meta['source_sha256']==source_hash and meta['tmc_permutations_per_seed']==100 and meta['tmc_seeds']==[101]
        key=(meta['datasets'][0],meta['outer_fold_index']);assert key not in keys;keys.add(key)
        res=frame('/fold_results.csv');pred=frame('/outer_predictions.csv');tim=frame('/fold_timings.csv')
        alloc=frame('/final_allocations.csv');selected=frame('/selected_rows.csv')
        test=set(alloc.loc[alloc.role=='test','patient_row']);val=set(alloc.loc[alloc.role=='valuation','patient_row'])
        assert alloc.patient_row.is_unique
        for method,g in selected.groupby('method'):
            assert not set(g.patient_row)&test
            if method!='TML':assert not set(g.patient_row)&val
        for _,g in frame('/inner_allocations.csv').groupby('inner_fold'):assert g.patient_row.is_unique and not set(g.patient_row)&test
        for (method,rep),g in pred.groupby(['method','repeat']):
            assert set(g.patient_row)==test and g.patient_row.is_unique
            row=res[(res.method==method)&(res['repeat']==rep)].iloc[0];m=binary_metrics(g.y_true,g.y_probability)
            assert np.allclose([m[k] for k in METRICS],row[METRICS].to_numpy(float),rtol=1e-6,atol=1e-7)
            checked+=4
        for name,raw in files.items():
            if name.endswith('_tmc_trace.csv'):
                trace=pd.read_csv(io.BytesIO(raw));assert len(trace)==100 and np.allclose(trace.marginal_sum,trace.terminal_utility)
        results.append(res);timings.append(tim)
    res=pd.concat(results,ignore_index=True);tim=pd.concat(timings,ignore_index=True)
    assert keys=={(d,k) for d in DATASETS for k in range(10)}
    fm=res.groupby(['dataset','fold','method'],as_index=False)[METRICS].mean()
    summary=fm.groupby(['dataset','method'])[METRICS].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
    saved=pd.read_csv(ROOT/'results/benchmark/summary.csv');idx=['dataset','method'];cols=[m+'_'+s for m in METRICS for s in ['mean','std']]
    compare(summary.set_index(idx).sort_index(),saved.set_index(idx).sort_index(),cols)
    rows=[]
    for m in METRICS:
        wide=summary[summary.method.isin(['SEPM','TMC'])].pivot(index='dataset',columns='method',values=m+'_mean');diff=wide.SEPM-wide.TMC
        rows.append(dict(metric=m,n_datasets=4,mean_SEPM_minus_TMC=diff.mean(),**exact_signed_rank(diff)))
    stats=pd.DataFrame(rows);stats['p_holm_four_metrics']=holm(stats.p_exact)
    assert np.allclose(stats.p_exact,.125) and np.allclose(stats.p_holm_four_metrics,.5)
    save(summary,'benchmark_summary.csv');save(res,'benchmark_fold_results.csv');save(tim,'benchmark_timings.csv');save(stats,'benchmark_wilcoxon_holm.csv')
    lookup=summary.set_index(idx);table=[]
    for ds,(label,_) in DATASETS.items():
        for method in ['SEPM','TMC']:
            row={'Dataset':label,'Method':method};a=lookup.loc[(ds,method)]
            for m in METRICS:row[m]=f'{a[m+"_mean"]:.3f} ± {a[m+"_std"]:.3f}'
            table.append(row)
    save(pd.DataFrame(table),'Table4_benchmark_performance.csv')
    means=tim.groupby(['dataset','method']).total_pipeline_seconds.mean();table=[]
    for ds,(label,_) in DATASETS.items():
        a,b=means.loc[(ds,'SEPM')],means.loc[(ds,'TMC')]
        table.append(dict(Dataset=label,SEPM_seconds_per_fold=a,TMC_seconds_per_fold=b,TMC_over_SEPM=b/a))
    save(pd.DataFrame(table),'Table5_benchmark_runtime.csv')
    for m in ['ece','brier']:
        w=summary[summary.method.isin(['SEPM','TMC'])].pivot(index='dataset',columns='method',values=m+'_mean');assert (w.SEPM<w.TMC).all()
    return dict(jobs=40,prediction_metric_cells=checked,source_hash_matched=True,metrics_tested=4)

def simulation():
    folder=ROOT/'results/simulation';raw=pd.read_csv(folder/'repetition_results.csv');fm=pd.read_csv(folder/'repetition_means.csv')
    summary=pd.read_csv(folder/'summary.csv');paired=pd.read_csv(folder/'paired_differences.csv');ret=pd.read_csv(folder/'training_retention.csv')
    keys=['scenario','n_train','target_prevalence','repetition','method','test_group'];gkeys=[k for k in keys if k!='repetition']
    computed=raw.groupby(keys)[METRICS].mean().sort_index();saved=fm.set_index(keys).sort_index();assert computed.index.equals(saved.index);compare(computed,saved,METRICS)
    for m in METRICS:
        agg=computed.groupby(gkeys)[m].agg(['mean','std','count']);actual=summary.set_index(gkeys).sort_index();agg=agg.reindex(actual.index)
        for col in ['mean','std','count']:assert np.allclose(actual[m+'_'+col],agg[col],atol=1e-10,equal_nan=True)
        assert np.allclose(actual[m+'_mcse'],agg['std']/np.sqrt(agg['count']),atol=1e-10,equal_nan=True)
    for _,row in paired.iterrows():
        sub=fm[(fm.scenario==row.scenario)&(fm.test_group==row.test_group)]
        w=sub.pivot(index='repetition',columns='method',values=row.metric);other=row.comparison.removeprefix('SEPM minus ');diff=(w.SEPM-w[other]).dropna()
        assert np.allclose([len(diff),diff.mean(),diff.std(ddof=1),diff.std(ddof=1)/np.sqrt(len(diff))],[row.n_repetitions,row.mean_difference,row.sd_difference,row.mcse_difference],atol=1e-10)
    full=fm[(fm.method=='SEPM')&(fm.test_group=='full')]
    assert len(full)==2000 and full.groupby(['n_train','target_prevalence']).size().eq(100).all()
    assert len(full[['n_train','target_prevalence']].drop_duplicates())==20
    low=fm[(fm.method=='SEPM')&(fm.test_group=='low_shap')];assert low.f1.notna().sum()==1750
    source_hash=hashlib.sha256((ROOT/'simulation/sepm_simulation.py').read_bytes()).hexdigest()
    audit=json.loads((folder/'audit_summary.json').read_text());assert audit['source_versions']==[source_hash]
    assert audit['completed_repetitions']==2000 and audit['missing_repetitions_no_archive_or_result']==0
    save(summary,'simulation_summary.csv');save(paired,'simulation_paired_differences.csv')
    lookup=summary[summary.test_group=='full'].set_index(['n_train','target_prevalence','method'])
    retention=ret[ret.method=='SEPM'].groupby(['n_train','target_prevalence']).retained_fraction.mean()
    table=[]
    for n in [100,200,500,1000,5000]:
        for prev in [.1,.2,.3,.35]:
            row={'n':n,'Prevalence':prev}
            for method in METHODS:
                a=lookup.loc[(n,prev,method)];row[method]=f'{a.f1_mean:.3f} ± {a.f1_std:.3f}'
            row['Retained']=f'{100*retention.loc[(n,prev)]:.1f}%';table.append(row)
    save(pd.DataFrame(table),'Table6_simulation_F1.csv')
    table=[]
    for group,label in [('full','FTG'),('high_shap','HSTG'),('low_shap','LSTG')]:
        for method in ['SEPM','SEPM-RAND','TML']:
            g=fm[(fm.test_group==group)&(fm.method==method)];row=dict(Group=label,Method=method,Valid_n=int(g.f1.notna().sum()))
            row.update({m:g[m].mean() for m in METRICS});table.append(row)
    save(pd.DataFrame(table),'Table7_simulation_groups.csv')
    return dict(raw_records=len(raw),repetition_means=len(fm),scenarios=20,repetitions=2000,valid_low_SHAP_repetitions=1750,summary_rows=len(summary),paired_rows=len(paired),source_hash_matched=True)

def main():
    OUT.mkdir(exist_ok=True)
    print('Verifying final clinical results...',flush=True);clinical,_=main_clinical()
    print('Verifying 40 completed common-protocol benchmark jobs...',flush=True);tmc=benchmark()
    print('Verifying 20-scenario simulation summaries...',flush=True);sim=simulation()
    report=dict(status='passed',clinical=clinical,benchmark=tmc,simulation=sim,model_training_rerun=False)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2));print('Reproduced tables and statistics:',OUT)

if __name__=='__main__':
    main()
