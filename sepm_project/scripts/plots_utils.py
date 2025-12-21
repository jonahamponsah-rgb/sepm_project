import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

import seaborn as sns
from sklearn.calibration import calibration_curve

# =====================
# Class Distribution Plot Function
# =====================
def plot_class_distribution(
    df: pd.DataFrame,
    target_col: str = "label",
    title: str = "Class Distribution",
    outbase: str = "class_distribution",
    save: bool = False,
    save_dir: str = "sepm/figures",
):
    """
    Plots and optionally saves the class distribution as a bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the target column.
    target_col : str, default="label"
        Name of the column with class labels.
    title : str
        Title for the plot.
    outbase : str
        Base filename for saving plots.
    save : bool
        If True, saves the plot.
    save_dir : str
        Directory where plots will be saved.
    """


    class_counts = df[target_col].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax, palette="pastel")

    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)

    for i, count in enumerate(class_counts.values):
        ax.text(i, count + 0.5, str(count), ha="center", va="bottom", fontsize=10)

    fig.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{outbase}.png"), dpi=300)
        fig.savefig(os.path.join(save_dir, f"{outbase}.pdf"))

    plt.show()


# =====================
# Top Percentage Distribution Function
# =====================
def plot_top_percentage_distribution_bestp(
    all_nested_cv_results,
    title: str = None,
    outbase: str = "top_percentage_distribution",
    save: bool = False,
    save_dir: str = "sepm/figures",
):
    """
    Plots and optionally saves a boxplot of the selected top percentage
    (best-p per fold) for SEPM and SEPM-RAND.

    Parameters
    ----------
    all_nested_cv_results : pd.DataFrame
        Must contain columns: 'Model', 'Fold', 'Method', 'Top Percentage', 'F1 Score'.
    title : str, optional
        Title for the plot.
    outbase : str, default="top_percentage_distribution"
        Base filename for saving plots.
    save : bool, default=False
        If True, saves the plot.
    save_dir : str, default="sepm/figures"
        Directory where plots will be saved.
    """

    df = all_nested_cv_results.copy()
    df = df[df["Method"].isin(["SEPM", "SEPM-RAND"])].copy()
    df = df[df["Top Percentage"].notna()].copy()
    df["Top Percentage"] = df["Top Percentage"].astype(float)

    # pick best p per fold per method/model (max F1; tie -> smaller p)
    def pick_best(g):
        max_f1 = g["F1 Score"].max()
        cand = g[g["F1 Score"] == max_f1].sort_values("Top Percentage", ascending=True)
        return cand.iloc[[0]]

    best_df = (
        df.groupby(["Model", "Fold", "Method"], group_keys=False)
          .apply(pick_best)
          .reset_index(drop=True)
    )

    best_df["Top Percentage"] = best_df["Top Percentage"] * 100.0  # convert to %

    data = [
        best_df.loc[best_df["Method"] == m, "Top Percentage"].values
        for m in ["SEPM", "SEPM-RAND"] if m in best_df["Method"].unique()
    ]
    labels = [m for m in ["SEPM", "SEPM-RAND"] if m in best_df["Method"].unique()]

    fig, ax = plt.subplots()
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel("Selected Top Percentage (%)")
    ax.set_title(title or "Cervical Cancer: Selected Top Percentage (Best-p per Fold)")
    ax.grid(True)

    fig.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{outbase}.png"), dpi=300)
        fig.savefig(os.path.join(save_dir, f"{outbase}.pdf"))

    plt.show()

# =====================
# Reliability Diagram Function
# =====================
def plot_reliability_diagram(
        df: pd.DataFrame,
        n_bins: int = 10,
        title: str = "Reliability Diagram",
        outbase: str = "reliability_diagram",
        save: bool = False,
        save_dir: str = "sepm/figures",
):
    """
    Plots a reliability diagram comparing calibration across methods.
    Aggregates predictions from all folds for each method.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'Method', 'y_true', 'y_prob'.
    n_bins : int
        Number of bins for calibration curve.
    title : str
        Title for the plot.
    outbase : str
        Base filename for saving plots.
    save : bool
        If True, saves the plot.
    save_dir : str
        Directory where plots will be saved.
    """

    methods = df["Method"].unique()
    fig, ax = plt.subplots(figsize=(8, 6))

    for method in methods:
        subset = df[df["Method"] == method]
        y_true_all = np.concatenate(subset["y_true"].values)
        y_prob_all = np.concatenate(subset["y_prob"].values)

        prob_true, prob_pred = calibration_curve(
            y_true_all, y_prob_all, n_bins=n_bins, strategy="uniform"
        )
        ax.plot(prob_pred, prob_true, marker="o", label=method)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Event Rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{outbase}.png"), dpi=300)
        fig.savefig(os.path.join(save_dir, f"{outbase}.pdf"))

    plt.show()




def _ci_to_yerr(mean, ci_low, ci_high):
    err_low = np.asarray(mean) - np.asarray(ci_low)
    err_high = np.asarray(ci_high) - np.asarray(mean)
    return np.vstack([err_low, err_high])


def _grouped_bar(
    models,
    methods,
    values_by_method,
    yerr_by_method=None,
    ylabel="",
    title="",
    legend_loc="lower right",
    legend_bbox=None,
    legend_ncol=None,
    figsize=(10, 5),
    capsize=5,
):
    x = np.arange(len(models))
    width = 0.22 if len(methods) >= 3 else 0.30

    plt.figure(figsize=figsize)

    for i, method in enumerate(methods):
        vals = values_by_method[method]
        offset = (i - (len(methods) - 1) / 2) * width
        if yerr_by_method is not None and method in yerr_by_method and yerr_by_method[method] is not None:
            plt.bar(x + offset, vals, width=width, yerr=yerr_by_method[method], capsize=capsize, label=str(method))
        else:
            plt.bar(x + offset, vals, width=width, label=str(method))

    plt.xticks(x, models, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    legend_kwargs = {"title": "Method", "frameon": False, "loc": legend_loc}
    if legend_bbox is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox
    if legend_ncol is not None:
        legend_kwargs["ncol"] = legend_ncol
    plt.legend(**legend_kwargs)
    plt.tight_layout()


def plot_grouped_metrics_with_ci(
    df: pd.DataFrame,
    out_dir: str | Path,
    dataset_name: str = "Dataset",
    method_order=("TML", "SEPM", "SEPM-RAND"),
    legend_loc="lower right",
    legend_bbox=None,
    legend_ncol=None,
    save_png=True,
    save_pdf=True,
    dpi=300,
    figsize=(8, 6),
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    present_methods = df["Method"].astype(str).unique().tolist()
    methods = [m for m in method_order if m in present_methods]
    if not methods:
        raise ValueError("No methods from method_order present in the DataFrame.")

    models = df["Model"].astype(str).unique().tolist()

    def _plot_metric(metric_name: str, ylabel: str):
        mean_col = f"Mean {metric_name}"
        low_col = f"CI Low {metric_name}"
        high_col = f"CI High {metric_name}"

        required = {"Method", "Model", mean_col, low_col, high_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for {metric_name}: {sorted(missing)}")

        sub = df[["Method", "Model", mean_col, low_col, high_col]].copy()
        pivot_val = sub.pivot(index="Model", columns="Method", values=mean_col).reindex(models)
        pivot_low = sub.pivot(index="Model", columns="Method", values=low_col).reindex(models)
        pivot_high = sub.pivot(index="Model", columns="Method", values=high_col).reindex(models)

        values_by_method = {m: pivot_val[m].values for m in methods}
        yerr_by_method = {
            m: _ci_to_yerr(pivot_val[m].values, pivot_low[m].values, pivot_high[m].values)
            for m in methods
        }

        _grouped_bar(
            models=models,
            methods=methods,
            values_by_method=values_by_method,
            yerr_by_method=yerr_by_method,
            ylabel=ylabel,
            title=f"{dataset_name}: {metric_name} (Mean with 95% CI)",
            legend_loc=legend_loc,
            legend_bbox=legend_bbox,
            legend_ncol=legend_ncol,
            figsize=figsize,
        )

        base_name = f"{dataset_name.lower().replace(' ', '_')}_{metric_name.lower().replace(' ', '_')}_grouped_ci"
        out_png = out_dir / f"{base_name}.png"
        out_pdf = out_dir / f"{base_name}.pdf"

        if save_png:
            plt.savefig(out_png, dpi=dpi, bbox_inches="tight" if legend_bbox else None)
        if save_pdf:
            plt.savefig(out_pdf, bbox_inches="tight" if legend_bbox else None)
        plt.close()

    for metric, ylabel in [
        ("F1 Score", "F1 Score"),
        ("Balanced Accuracy", "Balanced Accuracy"),
        ("Brier Score", "Brier Score"),
        ("ECE", "Expected Calibration Error (ECE)"),
    ]:
        mean_col = f"Mean {metric}"
        low_col = f"CI Low {metric}"
        high_col = f"CI High {metric}"
        if all(c in df.columns for c in [mean_col, low_col, high_col]):
            _plot_metric(metric, ylabel)






