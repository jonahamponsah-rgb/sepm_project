
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import statsmodels.stats.multitest as smm


# Step 1: Select the best configuration per fold/method/model
def select_best_per_fold(df, metric="F1 Score"):
    """Select the best-performing configuration per fold/method/model."""
    best_rows = []
    for (_, model, method), group in df.groupby(["Fold", "Model", "Method"]):
        max_val = group[metric].max()
        best_group = group[group[metric] == max_val]
        best_row = best_group.loc[best_group["Top Percentage"].idxmin()]
        best_rows.append(best_row)
    return pd.DataFrame(best_rows)

# Step 2: Paired comparisons across folds within models

def run_friedman_posthoc(best_df, metric="F1 Score", correction="holm"):
    """
    Run Friedman omnibus test, Conover posthoc with Holm correction,
    and compute average ranks for any chosen metric.

    Parameters
    ----------
    best_df : pd.DataFrame
        Must contain columns: 'Fold', 'Method', and the chosen metric
    metric : str
        Column name of the performance metric to analyze (default 'F1 Score')
    correction : str
        Multiple testing correction method (default 'holm')

    Returns
    -------
    results : dict
        {
            'friedman': (statistic, pvalue),
            'posthoc_raw': pd.DataFrame of raw Conover p-values,
            'posthoc_corrected': pd.DataFrame of corrected p-values,
            'avg_rank': pd.Series of average ranks per method
        }
    """
    if metric not in best_df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame columns.")

    # --- Friedman omnibus test ---
    methods = best_df["Method"].unique()
    score_lists = [best_df.loc[best_df["Method"]==m, metric].tolist() for m in methods]
    friedman_res = friedmanchisquare(*score_lists)

    # --- Unique df: one score per Fold × Method ---
    df_unique = best_df.groupby(["Fold","Method"], as_index=False)[metric].mean()

    # --- Conover posthoc (wide format) ---
    pivot = df_unique.pivot(index="Fold", columns="Method", values=metric)
    x = pivot.to_numpy()
    posthoc_raw = sp.posthoc_conover_friedman(x)
    posthoc_raw.index = pivot.columns
    posthoc_raw.columns = pivot.columns

    # --- Holm correction ---
    pvals = posthoc_raw.values[np.triu_indices_from(posthoc_raw, k=1)]
    _, pvals_corrected, _, _ = smm.multipletests(pvals, method=correction)

    posthoc_corrected = posthoc_raw.copy()
    posthoc_corrected.values[np.triu_indices_from(posthoc_raw, k=1)] = pvals_corrected
    posthoc_corrected.values[np.tril_indices_from(posthoc_raw, -1)] = pvals_corrected

    # --- Average ranks ---
    ranks = df_unique.groupby("Fold")[metric].rank(pct=True)
    avg_rank = ranks.groupby(df_unique["Method"]).mean()

    return {
        "friedman": (friedman_res.statistic, friedman_res.pvalue),
        "posthoc_raw": posthoc_raw,
        "posthoc_corrected": posthoc_corrected,
        "avg_rank": avg_rank
    }

def save_posthoc_plots(results, cd_path="cd_diagram.png"):
    """
    Save Conover/Nemenyi posthoc sign plot and critical difference diagram as image files.

    Parameters
    ----------
    results : dict
        Output from run_friedman_posthoc(), must contain keys:
        - 'posthoc': pd.DataFrame of pairwise p-values
        - 'avg_rank': pd.Series of average ranks per method
    cd_path : str
        File path to save the critical difference diagram (default 'cd_diagram.png')
    """

    # --- Critical difference diagram ---
    fig = plt.figure(figsize=(8, 6), dpi=300)
    plt.title("Critical difference diagram of average score ranks")
    sp.critical_difference_diagram(results["avg_rank"], results["posthoc"])
    fig.savefig(cd_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


