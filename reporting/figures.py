"""Generate Figures 2–6 from the verified numerical tables."""
from pathlib import Path
import runpy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduced/figures'
COLORS={'SEPM':'#2166A5','SEPM-RAND':'#DD7A21','TML':'#2C9A3D','TMC':'#A95C1E'}

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=300)
    fig.savefig(OUT/(name+'.svg'))
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    required=ROOT/'reproduced/verification.json'
    if not required.exists():raise FileNotFoundError('Run reproduce_results.py before make_figures.py.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    summary=pd.read_csv(ROOT/'reproduced/clinical_summary.csv').set_index(['dataset','model','method'])
    datasets=['Cervical cancer','Breast recurrence','Pima diabetes','Haberman']
    models=['Logistic Regression','SVM','XGBoost','GBM','Random Forest']
    delta=np.array([[100*(summary.loc[(d,m,'SEPM'),'f1_mean']-summary.loc[(d,m,'TML'),'f1_mean']) for d in datasets] for m in models])
    retention=np.array([[100*summary.loc[(d,m,'SEPM'),'retained_fraction'] for d in datasets] for m in models])
    fig,axes=plt.subplots(1,2,figsize=(9,4.2),layout='constrained')
    for ax,values,panel,cmap,limits in [(axes[0],delta,'a','RdBu',(-22,22)),(axes[1],retention,'b','Blues',(35,100))]:
        im=ax.imshow(values,cmap=cmap,vmin=limits[0],vmax=limits[1],aspect='auto')
        ax.set_xticks(range(4),['Cervical','Breast','Diabetes','Haberman'],rotation=25,ha='right');ax.set_yticks(range(5),['LR','SVM','XGB','GBM','RF'])
        ax.text(-.07,1.04,panel,transform=ax.transAxes,fontweight='bold')
        for i in range(5):
            for j in range(4):
                rgb=im.cmap(im.norm(values[i,j]));light=sum(c*v for c,v in zip([.2126,.7152,.0722],rgb[:3]))
                ax.text(j,i,f'{values[i,j]:+.2f}' if panel=='a' else f'{values[i,j]:.1f}%',ha='center',va='center',color='white' if light<.5 else 'black',fontsize=9)
        ax.set_xlabel('F1 difference (SEPM minus TML; percentage points)' if panel=='a' else 'Training observations retained')
    save(fig,'Fig2_clinical')

    bench=pd.read_csv(ROOT/'reproduced/benchmark_summary.csv').set_index(['dataset','method'])
    times=pd.read_csv(ROOT/'reproduced/benchmark_timings.csv').groupby(['dataset','method']).total_pipeline_seconds.mean()
    ds=['cervical','breast','diabetes','haberman'];x=np.arange(4)
    fig,axes=plt.subplots(1,2,figsize=(9,4.2),layout='constrained')
    for method,offset,hatch in [('SEPM',-.17,'///'),('TMC',.17,'...')]:
        axes[0].bar(x+offset,[bench.loc[(d,method),'f1_mean'] for d in ds],width=.32,color=COLORS[method],label=method,hatch=hatch,edgecolor='black',linewidth=.5)
        axes[1].bar(x+offset,[times.loc[(d,method)] for d in ds],width=.32,color=COLORS[method],label=method,hatch=hatch,edgecolor='black',linewidth=.5)
    axes[0].set_ylim(0,1);axes[0].set_ylabel('Mean outer-fold F1');axes[1].set_yscale('log');axes[1].set_ylabel('Pipeline seconds per fold (log scale)')
    for ax,panel in zip(axes,['a','b']):
        ax.set_xticks(x,['Cervical','Breast','Diabetes','Haberman'],rotation=25,ha='right');ax.legend(frameon=False);ax.text(-.08,1.04,panel,transform=ax.transAxes,fontweight='bold')
    save(fig,'Fig3_benchmark')

    sim=pd.read_csv(ROOT/'reproduced/simulation_summary.csv');full=sim[sim.test_group=='full']
    fig,axes=plt.subplots(2,2,figsize=(9,7),layout='constrained')
    sizes=[100,200,500,1000,5000]
    for ax,prev in zip(axes.flat,[.1,.2,.3,.35]):
        for method,marker in [('SEPM','o'),('SEPM-RAND','s'),('TML','^')]:
            g=full[(full.target_prevalence==prev)&(full.method==method)].set_index('n_train').reindex(sizes)
            ax.plot(range(5),g.f1_mean,label=method,color=COLORS[method],marker=marker,markersize=4)
        ax.set_xticks(range(5),[f'{n:,}' for n in sizes]);ax.set_xlabel('Training sample size');ax.set_ylabel('Mean F1 score');ax.set_title(f'Target prevalence: {prev:.0%}');ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=3,frameon=False)
    save(fig,'Fig4_simulation_size')

    grouped=pd.read_csv(ROOT/'reproduced/Table7_simulation_groups.csv');groups=['FTG','HSTG','LSTG']
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    for method,marker in [('SEPM','o'),('SEPM-RAND','s'),('TML','^')]:
        g=grouped[grouped.Method==method].set_index('Group').reindex(groups)
        ax.plot(range(3),g.f1,label=method,color=COLORS[method],marker=marker,markersize=7)
    ax.set_xticks(range(3),['Full test group (FTG)','High-SHAP (HSTG)','Low-SHAP (LSTG)']);ax.set_ylabel('Mean F1 score');ax.grid(axis='y',alpha=.25)
    counts=grouped[grouped.Method=='SEPM'].set_index('Group').Valid_n
    ax.text(.01,1.02,'Valid repetitions: '+ '; '.join(f'{g} = {counts[g]:,}' for g in groups),transform=ax.transAxes,fontsize=10)
    ax.legend(frameon=False,ncols=3,loc='lower left')
    save(fig,'Fig5_simulation_groups')

    paired=pd.read_csv(ROOT/'reproduced/simulation_paired_differences.csv');pairs=paired[(paired.comparison=='SEPM minus TML')&(paired.metric=='f1')]
    order=[(n,p) for n in sizes for p in [.1,.2,.3,.35]];y=np.arange(20)
    fig,axes=plt.subplots(1,3,figsize=(10,8),layout='constrained',sharey=True)
    for ax,group,title,panel in zip(axes,['full','high_shap','low_shap'],['Full test group','High-SHAP group','Low-SHAP group'],['a','b','c']):
        g=pairs[pairs.test_group==group].set_index(['n_train','target_prevalence']).reindex(order)
        ax.errorbar(g.mean_difference,y,xerr=g.mcse_difference,fmt='o',color=COLORS['SEPM'],markersize=4,capsize=2)
        ax.axvline(0,color='#999999',lw=.8);ax.set_title(title);ax.set_xlabel('F1 difference\n(SEPM minus TML)');ax.set_xlim(-.01,.19);ax.set_xticks([0,.05,.10,.15]);ax.grid(axis='x',alpha=.2);ax.text(-.1,1.04,panel,transform=ax.transAxes,fontweight='bold')
        for split in [3.5,7.5,11.5,15.5]:ax.axhline(split,color='#dddddd',lw=.7)
    axes[0].set_yticks(y,[f'{n:,} / {p:.0%}' for n,p in order]);axes[0].invert_yaxis();axes[0].set_ylabel('Training size / target prevalence')
    save(fig,'Fig6_simulation_differences')
    print('Generated Figures 2–6:',OUT)

if __name__=='__main__':
    main()
