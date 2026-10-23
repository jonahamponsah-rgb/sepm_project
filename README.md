# SHAP Enhanced Predictive Modeling for Clinical Machine Learning

Reproducibility package for SEPM by Jonah K. Amponsah, Jeffrey A. Thompson and Shellie D. Ellis. Repository: https://github.com/jonahamponsah-rgb/sepm_project

This package contains the final clinical programs, the exact CHTC benchmark and simulation implementations, saved results, and code to reconstruct manuscript tables and statistical comparisons. Settings remain inside the scripts; a separate configuration folder is unnecessary.

## Start here — reproduce reported results without retraining

Install `requirements.txt` in a Python environment, open `reproduce_results.py` in DataSpell/PyCharm and run the whole file. It uses only NumPy and pandas. Alternatively:

```bash
python -m pip install -r requirements.txt
python reproduce_results.py
```

Outputs go to `reproduced/`; archived results are never overwritten. The program checks clinical probabilities against four saved metrics, selected proportions against mean validation F1, all 40 benchmark jobs against their predictions and allocations, and simulation summaries against 2,000 repetition records. It recreates the seven manuscript data tables (Tables 1–7), supplementary metric tables, main Friedman/three-pair Wilcoxon–Holm results, and all four SEPM–TMC Wilcoxon–Holm comparisons. Verify `reproduced/verification.json` after execution.

For data figures, install `requirements-figures.txt`, run `make_figures.py`, and find Figures 2–6 in `reproduced/figures/`. Figure 1 is a schematic supplied with its editable source under `reporting/`; run `reporting/draw_workflow.py` to regenerate it. All packaged figures are also supplied under `results/figures/`; plotting does not recompute model predictions. Figure styles may differ from final Word formatting while numerical content is reproduced.

## Retrain the clinical analyses

The saved clinical runs used Python 3.13.11 and the versions in `requirements-clinical.txt`. Use a separate environment from the cluster benchmark. Obtain the data described in `data/README.md`; existing local files can be copied and checksum-verified by editing `SOURCE_DIRECTORY` and running `prepare_data.py`.

Run any of these directly from your IDE:

* `clinical/sepm_cervical.py`
* `clinical/sepm_breast.py`
* `clinical/sepm_diabetes.py`
* `clinical/sepm_haberman.py`

Each runs five classifiers and saves new results under `reruns/clinical/<dataset>/`. The scripts refuse an already populated output directory: choose another `OUTPUT` or move the prior rerun yourself. The packaged `results/` are immutable reference records.

Clinical settings: seed 30; 10 outer folds and 5 inner folds; three SEPM-RAND draws; candidate proportions 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 1.00. Hyperparameters are tuned first and fixed during proportion selection. Maximum mean inner-validation F1 determines the proportion; exact ties favor larger proportions. Percentile ties are retained and an outcome-class safeguard is applied. Proportion-selection F1 and outer classification use probability cutoff 0.5; baseline GridSearchCV retains native-prediction F1. The scoring distinction matters for SVM. No threshold optimization is performed.

## Common-protocol SEPM–TMC benchmark

Use the recorded package versions in `requirements-cluster.txt` (historical CHTC Python 3.11 context). Run `run_benchmark.py` from your IDE for all four datasets, or use `benchmark/chtc_fold_runner.py` for one of the fixed outer folds. The driver overrides the original module's pilot defaults to 100 permutations, seed 101 and all ten folds. The final benchmark uses an 80% candidate pool and 20% held-out valuation reserve within training, XGBoost, matched partitions, independently optimized retained proportions, and the recorded TMC truncation settings. Auxiliary methods remain in the original implementation for provenance; Tables 4–5 compare SEPM and TMC.

Exact Wilcoxon tests use four paired dataset mean scores, not 40 folds as independent units. Tests cover F1, balanced accuracy, ECE and Brier, with Holm adjustment across four metrics. All four have exact p = 0.125 and adjusted p = 0.500. This establishes neither superiority nor equivalence. Pipeline time is summarized separately from scheduler job duration.

## Simulation

Run `run_simulation.py` from your IDE or follow `cluster/README.md`. It evaluates five training sizes (100, 200, 500, 1,000, 5,000), four prevalences (10%, 20%, 30%, 35%) and 100 repetitions per scenario, with 5,000 independent test observations each. The batch runner reduces inner folds to the minority count when necessary without redrawing datasets. FTG and HSTG have 2,000 valid repetitions; empty LSTG cases leave 1,750. Three random draws are averaged within a repetition before comparison. Test SHAP partitions are model-defined and do not represent independently defined clinical subgroups.

## Contents

| Location | Contents |
|---|---|
| `clinical/` | Four final clinical experiment programs |
| `benchmark/` | Hash-matched CHTC program and original fold/aggregation utilities |
| `simulation/` | Hash-matched simulation program and batch/aggregation utilities |
| `reporting/` | Verification, exact statistics, tables and figure generation |
| `cluster/` | Job lists, shell wrappers, HTCondor templates, bundle builder and reconstructed container recipe |
| `results/clinical/` | Final folds, inner selection, predictions, timing, summaries and original metadata |
| `results/benchmark/` | Forty original fold archives, combined results and timing, scheduler records |
| `results/simulation/` | Complete combined final simulation records and completeness audit |
| `results/reported/` | Recomputed Tables 1–7, supplementary metric tables and final statistics |
| `results/figures/` | Verified generated figures and the workflow schematic |
| `data/` | Dataset access and expected original input hashes |
| `provenance/` | Source origins, hashes, package-setting notes and representative run metadata |

## Validation and limitations

`PACKAGE_VALIDATION.json` records the checks performed when assembling this package. Saved-result verification is separate from rerunning training. Historical model training and all CHTC jobs were not rerun during packaging. Exact package versions differ between the local clinical and cluster environments. The original cluster image/build recipe was unavailable; the included recipe is a reconstruction and is labelled accordingly. Detailed simulation predictions and training audits across 720 compressed jobs are not embedded in this compact GitHub package; the combined repetition results and audit files reproduce the reported simulation summaries, and full execution regenerates detailed files. Original CHTC simulation bundles can be shared through a repository release or data archive if needed.

The authors have left the software license undecided; see `LICENSE_STATUS.md`. Raw datasets retain their source licensing. No journal correspondence, author contact directories, obsolete analysis outputs, virtual environments or credentials are included.

## Upload

Extract the ZIP and upload the CONTENTS of this `sepm_project` directory to the repository root. Do not nest another `sepm_project` folder inside an existing package unless that is the intended repository layout. All included files are below GitHub's normal per-file limit. This package was prepared locally and has not been pushed or published.
