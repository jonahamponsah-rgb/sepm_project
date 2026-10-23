from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle

OUT=Path(__file__).resolve().parents[1]/'reproduced/figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(14,10),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1400);ax.set_ylim(0,1000);ax.axis('off')
navy='#172B45';muted='#52657B';line='#C8D3E0';blue='#2D6494';teal='#147D75';amber='#A56B20'
def txt(x,y,s,size=13,color=navy,weight='normal',ha='left',va='top',**kw):
 return ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,linespacing=1.45,**kw)
def box(x,y,w,h,fill='white',edge=line,lw=1.1,r=11):
 p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',facecolor=fill,edgecolor=edge,linewidth=lw);ax.add_patch(p);return p
def arrow(a,b,color=muted,lw=1.5,style='-|>',connectionstyle='arc3'):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=12,linewidth=lw,color=color,connectionstyle=connectionstyle,shrinkA=2,shrinkB=2))
def stage(x,w,n,title,color,fill):
 box(x,190,w,560,fill,edge=line,r=13)
 ax.add_patch(Circle((x+31,713),17,facecolor=color,edgecolor='none'))
 txt(x+31,713,str(n),14,'white','bold',ha='center',va='center');txt(x+60,726,title,17,weight='bold')
 ax.plot([x+20,x+w-20],[685,685],color=line,lw=.8)
def card(x,y,w,h,title,body,color=blue):
 box(x,y,w,h,'white',line,r=9)
 ax.plot([x,x],[y+15,y+h-15],color=color,lw=3,solid_capstyle='round')
 txt(x+17,y+h-15,title,13.5,weight='bold')
 txt(x+17,y+h-46,body,12.2,color=muted)

txt(44,964,'SEPM workflow',25,weight='bold')
txt(44,919,'Training-only observation selection with nested cross-validation',15,color=muted)
ax.plot([44,1355],[885,885],color=line,lw=1)
# Split the outer fold before any model fitting.
for x,w,title,body,fill,edge in [
(44,253,'Data and classifier','Predictors X, outcome y, grid Q','#F7F9FC',line),
(340,253,'10-fold outer CV','Stratified training / test splits','#F2F6FB',line),
(636,360,'Outer-training partition','All fitting, ranking and selection','#EDF7F5','#91C8C1'),
(1039,316,'Outer-test partition','Held out until final evaluation','#F5F6F8',line)]:
 box(x,790,w,75,fill,edge)
 txt(x+w/2,848,title,14,weight='bold',ha='center')
 txt(x+w/2,819,body,11.7,color=muted,ha='center')
arrow((300,827),(337,827));arrow((596,827),(632,827))
ax.plot([610,610,1197],[827,875,875],color='#91A1B4',lw=1.3)
arrow((1197,875),(1197,867),color='#91A1B4')
# Three aligned stages with a single, legible direction of travel.
stage(44,343,1,'Tune classifier',blue,'#F2F6FB')
stage(419,537,2,'Select retention proportion',amber,'#FCF8F0')
stage(988,367,3,'Fit and evaluate',teal,'#F0F8F6')
# Training partition feeds the training-only workflow.
ax.plot([815,815,215],[788,771,771],color=teal,lw=1.3)
arrow((215,771),(215,752),teal)
# Stage 1.
card(62,554,307,106,'Tune on outer-training data','Stratified 5-fold grid search\nwith the standard F1 scorer',blue)
arrow((215,551),(215,526),blue)
card(62,416,307,108,'Choose hyperparameters','Maximize mean validation F1',blue)
arrow((215,413),(215,388),blue)
card(62,295,307,91,'Freeze hyperparameters','Reuse during proportion selection',blue)
txt(215,255,'Outer-test data remain unseen',11.5,color=blue,weight='bold',ha='center')
ax.plot([371,403,403],[340,340,612],color=blue,lw=1.5)
arrow((403,612),(434,612),blue)
# Stage 2.
card(437,565,501,95,'Within each of 5 inner folds','Fit the tuned baseline using inner-training cases only',amber)
arrow((687,563),(687,548),amber)
card(437,459,501,87,'Compute training SHAP scores','',amber)
txt(454,485,r'$s_i = d^{-1}\,\sum_{j=1}^{d}|\phi_j(x_i)|$',12.2,color=muted,va='center')
arrow((687,457),(687,442),amber)
card(437,345,501,95,'For each candidate q in Q','Retain scores at or above the (1 − q) quantile;\nkeep cutoff ties and safeguard class presence',amber)
arrow((687,343),(687,328),amber)
card(437,236,501,90,'Refit on selected cases; validate','Refit preprocessing and classifier on the subset.\nRecord F1 on the entire inner-validation fold.',amber)
# Decision is a short strip outside the stacked cards; enough room for text.
# Bottom selection strip bridges stage 2 and final fit without crossing content.
box(419,104,537,66,'#FFF1D8','#D8B77E',r=10)
txt(687,155,'Select q* with highest mean inner-fold F1',13,weight='bold',ha='center')
txt(687,128,'Break exact F1 ties in favor of the larger proportion',11.5,color=muted,ha='center')
arrow((687,233),(687,173),amber)
# Stage 3.
card(1006,550,331,110,'Recompute the final ranking','Fit the tuned baseline on the full\nouter-training partition; compute SHAP.',teal)
arrow((1171,547),(1171,522),teal)
card(1006,403,331,117,'Apply selected q* and refit','Use the percentile and class safeguard.\nRefit preprocessing and classifier\non the selected observations.',teal)
arrow((1171,400),(1171,375),teal)
card(1006,239,331,134,'Evaluate once on the outer test','Predict every held-out observation.\nReport F1, balanced accuracy,\nECE and Brier score.',teal)
# q* enters final subset selection using a clear gap between panels.
ax.plot([958,972,972],[137,137,610],color=teal,lw=1.5)
arrow((972,610),(1003,610),teal)
# Held-out test route stays entirely outside the modeling cards.
ax.plot([1197,1374,1374],[788,788,306],color='#91A1B4',lw=1.3)
arrow((1374,306),(1340,306),color='#91A1B4')
# Compact footer, with the full grid supplied in a separate caption file.
box(44,18,1311,65,'#F7F9FC',line,r=10)
txt(64,68,'Repeat across all 10 outer folds',13.5,weight='bold')
txt(64,43,'Aggregate held-out metrics. Outer-test information never determines hyperparameters, SHAP rankings or the retained proportion.',11.8,color=muted)
for ext in ['svg','pdf','eps']:
 fig.savefig(OUT/f'SEPM_workflow.{ext}')
fig.savefig(OUT/'SEPM_workflow.png',dpi=220)
fig.savefig(OUT/'SEPM_workflow.tif',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)
(OUT/'Figure_caption.txt').write_text('SEPM workflow. Hyperparameters are tuned within the outer-training partition and frozen before training-only proportion selection. Each inner fold reconstructs preprocessing, the baseline model and SHAP ranking. After selection, ranking is recomputed on all outer-training data, and the final model is fitted on selected observations before evaluating all outer-test observations. Standard GridSearchCV F1 uses native class predictions; proportion-selection and outer-test F1 use a fixed probability cutoff of 0.5. Candidate retention grid Q = {0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 1.00}. Exact validation-F1 ties favor larger proportions. d denotes the number of predictors.\n',encoding='utf8')
print(OUT)
