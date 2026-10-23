"""Exact signed-rank and Holm calculations used by the manuscript reports."""
from math import exp
import numpy as np
import pandas as pd

def ranks(values):
    return pd.Series(np.asarray(values)).rank(method='average').to_numpy()

def holm(pvalues):
    p=np.asarray(pvalues,dtype=float);order=np.argsort(p)
    adjusted=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    result=np.empty(len(p));result[order]=adjusted
    return result

def exact_signed_rank(differences):
    """Two-sided exact sign-flip distribution, including tied absolute ranks.

    Zero differences are removed after rounding to 12 decimal places, matching
    the archived analysis. Integer-scaled midranks permit exact dynamic counting
    instead of allocating all 2**n sign assignments.
    """
    d=np.round(np.asarray(differences,dtype=float),12);d=d[d!=0]
    if not len(d):
        return dict(w_min=0.,p_exact=1.,n_nonzero=0,rank_biserial=0.,sign_assignments=1)
    r=ranks(abs(d));plus=float(r[d>0].sum());minus=float(r[d<0].sum())
    weights=np.rint(2*r).astype(int);counts={0:1}
    for weight in weights:
        nxt=dict(counts)
        for score,count in counts.items():nxt[score+int(weight)]=nxt.get(score+int(weight),0)+count
        counts=nxt
    observed=abs(plus-minus);total=int(weights.sum())
    tail=sum(count for score,count in counts.items() if abs(2*score-total)/2>=observed-1e-10)
    return dict(w_min=min(plus,minus),p_exact=tail/(2**len(d)),n_nonzero=len(d),
                rank_biserial=(plus-minus)/r.sum(),sign_assignments=2**len(d))

def friedman(scores):
    """Tie-corrected Friedman statistic, df=2 chi-square p and Kendall W."""
    x=np.asarray(scores,dtype=float);n,k=x.shape
    if k!=3:raise ValueError('This implementation reports the three-method benchmark only.')
    r=np.array([ranks(row) for row in x])
    ties=0
    for row in x:
        counts=np.unique(row,return_counts=True)[1];ties+=sum(counts**3-counts)
    correction=1-ties/(n*(k**3-k))
    q=(12*np.sum(r.sum(axis=0)**2)/(n*k*(k+1))-3*n*(k+1))/correction
    return dict(friedman_chi_square=float(q),df=2,p_chi_square_approximation=exp(-q/2),
                kendall_w=float(q/(n*(k-1))),n_tasks=n),r
