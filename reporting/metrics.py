"""The four manuscript metrics, implemented without model-training dependencies."""
import numpy as np

METRICS=['f1','balanced_accuracy','ece','brier']

def binary_metrics(y_true,probabilities):
    y=np.asarray(y_true).reshape(-1);p=np.asarray(probabilities,float).reshape(-1)
    if not len(y) or len(y)!=len(p) or not np.isin(y,[0,1]).all():raise ValueError('Invalid binary labels.')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid probabilities.')
    pred=p>=.5;tp=np.sum(pred&(y==1));fp=np.sum(pred&(y==0));fn=np.sum(~pred&(y==1));tn=np.sum(~pred&(y==0))
    recall=tp/(tp+fn) if tp+fn else np.nan;specificity=tn/(tn+fp) if tn+fp else np.nan
    bins=np.digitize(p,np.linspace(0,1,11)[1:-1],right=True)
    counts=np.bincount(bins,minlength=10);mask=counts>0
    sy=np.bincount(bins,weights=y,minlength=10);sp=np.bincount(bins,weights=p,minlength=10)
    return dict(f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
                balanced_accuracy=float(np.nanmean([recall,specificity])),
                ece=float(abs(sy[mask]-sp[mask]).sum()/len(y)),brier=float(np.mean((p-y)**2)))
