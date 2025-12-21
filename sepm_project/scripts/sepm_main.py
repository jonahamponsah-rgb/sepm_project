

from sepm_project.data.datasets import datasets
from sepm_project.scripts.classifiers import classifiers_dict
from sepm_project.scripts.sepm_with_ablation import run_pipeline_sepm_nested_cv


#######################################################################################################################
#%%
top_percentages = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 1]

all_results = run_pipeline_sepm_nested_cv(datasets=datasets, classifiers_dict=classifiers_dict,
                                         top_percentages=top_percentages,inner_splits=5, outer_splits=10)


