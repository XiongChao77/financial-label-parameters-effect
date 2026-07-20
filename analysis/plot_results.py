#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting script for the fixed-neutral-subsampling probe experiment (train.py).

This module is intentionally decoupled from model training/evaluation:
- train.py runs the experiment and persists all summary tables as CSV
  (plus a small meta.json) under `save_dir`.
- plot_results.py only READS those CSV/JSON files and produces all figures.

This means you can re-run / tweak plots without re-training anything:

    python plot_results.py --save_dir /path/to/BTCUSDT_1h/TBM/volatility

train.py also calls `generate_all_plots(...)` automatically right after it
finishes saving the summary tables, so plots are produced by default with
no extra steps.
"""

import os
import json
import argparse
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os,sys
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
sys.path.append(current_work_dir)
from data_process import common
from analysis import crossing_specificity
# The "model" name used for the random-prediction baseline everywhere in the
# summary tables (self_eval / financial). It is excluded from the normal
# per-model plotting loops and instead overlaid, mean-only, on every other
# model's figure.
BASELINE_MODEL_NAME = "RandomPredict"


def _split_baseline(df: pd.DataFrame, model_col: str = "model"):
    """Split a per-model summary dataframe into (real_models_df, baseline_df)."""
    if model_col not in df.columns:
        return df, df.iloc[0:0]
    is_baseline = df[model_col] == BASELINE_MODEL_NAME
    return df[~is_baseline].copy(), df[is_baseline].copy()


def _plot_baseline_mean(x, y, label="Baseline (Random)"):
    """
    Draw a mean-only baseline curve (no std band, no per-point annotation),
    with consistent styling everywhere it appears.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0:
        return
    order = np.argsort(x)
    plt.plot(
        x[order], y[order],
        marker="x", linewidth=2, linestyle="--", color="gray",
        alpha=0.85, label=label,
    )

def _scale_x(values, para_type: str):
    """Scale label-parameter X values for plotting: horizon values are divided by 10."""
    arr = np.asarray(values, dtype=float)
    if para_type == "horizon":
        return arr / 10.0
    return arr


def _overlay_crossings(crossings_df, series_filter=("empirical_vs_gaussian",),
                        curve_colors=None):
    """
    Draw a vertical dashed line at each unique crossing x0 from a
    crossing_specificity/LabelRatioCurveAnalyzer-style dataframe (columns:
    curve, series, crossing_order, x). crossings_df's x is already on the
    TRUE parameter scale (LabelRatioCurveAnalyzer sweeps volatility/horizon
    directly, no x10 naming quirk), so no _scale_x() call is needed here --
    it lines up with the already-scaled x-axis used by every plot below.

    series_filter: only these "series" values are drawn (default: only the
    pooled empirical_vs_gaussian series, to avoid cluttering the figure with
    up to 9 near-overlapping lines). Pass series_filter=None to draw every
    series (positive/negative/pooled) too.
    """
    if crossings_df is None or crossings_df.empty:
        return

    default_colors = {
        "ratio": "tab:purple",
        "first_derivative": "tab:brown",
        "second_derivative": "tab:red",
    }
    colors = {**default_colors, **(curve_colors or {})}

    curve_short_names = {
        "ratio": "Ratio",
        "first_derivative": "D1",
        "second_derivative": "D2",
    }

    df = crossings_df
    if series_filter is not None and "series" in df.columns:
        df = df[df["series"].isin(series_filter)]

    ax = plt.gca()
    ymin, ymax = ax.get_ylim()

    for _, row in df.iterrows():
        x0 = float(row["x"])
        curve = row.get("curve", "crossing")
        order = row.get("crossing_order", "")
        color = colors.get(curve, "black")
        short_name = curve_short_names.get(curve, curve)
        tag = f"{short_name} crossing #{order}\nx={x0:.2f}"

        plt.axvline(
            x0, color=color, linestyle=":", linewidth=1.3, alpha=0.75,
        )
        ax.text(
            x0, ymax, tag,
            rotation=270, color=color, fontsize=8,
            ha="right", va="top", alpha=0.9,
        )


def plot_cross_eval_heatmap(
    cross_df: pd.DataFrame,
    save_dir: str,
    metric: str,
    annotate_std: bool = True,
    para_type:str = "volatility",
) -> List[str]:
    """
    Plot cross-evaluation mean/std heatmaps.

    Expected input:
        cross_matrix_summary_df with columns:
        model, eval_mode, train_threshold, eval_threshold,
        {metric}_mean, {metric}_std
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    required_cols = [
        "model",
        "eval_mode",
        "train_threshold",
        "eval_threshold",
        mean_col,
        std_col,
    ]
    missing = [c for c in required_cols if c not in cross_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    saved_paths = []

    for (model_name, eval_mode), curr_df in cross_df.groupby(["model", "eval_mode"]):
        curr_save_dir = os.path.join(save_dir, str(model_name))
        os.makedirs(curr_save_dir, exist_ok=True)

        # Avoid pivot errors
        duplicated = curr_df.duplicated(
            ["train_threshold", "eval_threshold"],
            keep=False,
        )
        if duplicated.any():
            raise ValueError(
                f"Duplicate train/eval threshold pairs found for "
                f"model={model_name}, eval_mode={eval_mode}"
            )

        mean_pivot = (
            curr_df
            .pivot(
                index="train_threshold",
                columns="eval_threshold",
                values=mean_col,
            )
            .sort_index()
            .sort_index(axis=1)
        )

        std_pivot = (
            curr_df
            .pivot(
                index="train_threshold",
                columns="eval_threshold",
                values=std_col,
            )
            .reindex(index=mean_pivot.index, columns=mean_pivot.columns)
        )

        def add_mean_row_col(pivot: pd.DataFrame):
            values = pivot.values.astype(float)

            row_mean = np.nanmean(values, axis=1, keepdims=True)
            col_mean = np.nanmean(values, axis=0, keepdims=True)
            overall_mean = np.nanmean(values)

            values_with_mean = np.block([
                [values, row_mean],
                [col_mean, np.array([[overall_mean]])],
            ])

            row_labels = [f"{x:.1f}" for x in pivot.index] + ["Mean"]
            col_labels = [f"{x:.1f}" for x in pivot.columns] + ["Mean"]

            return values_with_mean, row_labels, col_labels

        mean_values, row_labels, col_labels = add_mean_row_col(mean_pivot)
        std_values, _, _ = add_mean_row_col(std_pivot)

        n_rows, n_cols = mean_values.shape
        fig_width = max(10, n_cols * 0.55)
        fig_height = max(8, n_rows * 0.45)

        # =====================================================
        # Mean heatmap
        # =====================================================
        vmin = np.nanmin(mean_pivot.values.astype(float))
        vmax = np.nanmax(mean_pivot.values.astype(float))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        im = ax.imshow(
            mean_values,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(row_labels)

        if para_type == 'volatility':
            label = "Threshold λ"
        elif para_type == 'horizon':
            label = "Bar horizon h"
        ax.set_xlabel(f"Eval Label {label}")
        ax.set_ylabel(f"Train Label {label}")
        ax.set_title(
            f"Cross Evaluation Mean {metric.upper()}\n"
            f"Model={model_name}, Eval={eval_mode}"
        )

        # Grid
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", linewidth=0.5, alpha=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        fontsize = 7 if max(n_rows, n_cols) > 20 else 8

        for i in range(n_rows):
            for j in range(n_cols):
                mean_val = mean_values[i, j]
                std_val = std_values[i, j]

                if np.isnan(mean_val):
                    text = ""
                elif annotate_std:
                    text = f"{mean_val:.5f}\n±{std_val:.5f}"
                else:
                    text = f"{mean_val:.5f}"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="black",
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        mean_out_path = os.path.join(
            curr_save_dir,
            f"cross_eval_{metric}_{eval_mode}_mean_heatmap.png",
        )
        plt.savefig(mean_out_path, dpi=220)
        plt.close()
        saved_paths.append(mean_out_path)

        # =====================================================
        # Std heatmap
        # =====================================================
        std_vmin = np.nanmin(std_pivot.values.astype(float))
        std_vmax = np.nanmax(std_pivot.values.astype(float))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        im = ax.imshow(
            std_values,
            aspect="auto",
            vmin=std_vmin,
            vmax=std_vmax,
        )

        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(row_labels)

        ax.set_xlabel("Eval Label Threshold λ")
        ax.set_ylabel("Train Label Threshold λ")
        ax.set_title(
            f"Cross Evaluation Std {metric.upper()}\n"
            f"Model={model_name}, Eval={eval_mode}"
        )

        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", linewidth=0.5, alpha=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(n_rows):
            for j in range(n_cols):
                val = std_values[i, j]
                text = "" if np.isnan(val) else f"{val:.3f}"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="black",
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        std_out_path = os.path.join(
            curr_save_dir,
            f"cross_eval_{metric}_{eval_mode}_std_heatmap.png",
        )
        plt.savefig(std_out_path, dpi=220)
        plt.close()
        saved_paths.append(std_out_path)

    return saved_paths

def plot_self_eval_mean_std(
    self_summary_df: pd.DataFrame,
    save_dir: str,
    metric: str = "macro_f1",
    para_type:str = "volatility",
    crossings_df: pd.DataFrame = None,
) -> list[str]:
    """
    Plot threshold-vs-metric curves for self evaluation.

    If `self_summary_df` contains an "eval_mode" column (balanced / raw),
    each eval_mode gets its own separate figure/file (e.g.
    self_eval_{metric}_balanced.png, self_eval_{metric}_raw.png);
    otherwise a single figure is produced (kept for backward compatibility).

    Rows with model == BASELINE_MODEL_NAME ("RandomPredict") are not plotted
    as their own model; instead they are overlaid on every other model's
    figure as a mean-only dashed baseline curve.

    crossings_df: optional dataframe from crossing_specificity.load_crossings()
    (columns curve/series/crossing_order/x). When given, each crossing x0 is
    drawn as a vertical dashed line so the crossing points can be visually
    compared against the performance curve directly.
    """

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    out_paths = []

    has_eval_mode = "eval_mode" in self_summary_df.columns

    real_df, baseline_df = _split_baseline(self_summary_df)

    for model_name, df_model in real_df.groupby("model", dropna=False):
        model_save_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)

        if has_eval_mode:
            groups = list(df_model.groupby("eval_mode", dropna=False))
        else:
            groups = [(None, df_model)]

        for eval_mode, df_one in groups:
            df_plot = df_one.sort_values("threshold").copy()

            x = df_plot["threshold"].to_numpy()
            y = df_plot[mean_col].to_numpy()
            y_std = df_plot[std_col].fillna(0).to_numpy()

            if len(x) == 0:
                continue

            train_size = int(df_plot["train_size"].iloc[0])
            test_size = int(df_plot["test_size"].iloc[0])
            total_runs = int(df_plot["n_runs"].max())

            info = (
                f"{total_runs} independent runs\n"
                f"train samples = {train_size}\n"
                f"test samples = {test_size}"
            )

            plt.figure(figsize=(10, 6))

            plt.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                label=f"{metric} mean",
            )

            plt.fill_between(
                x,
                y - y_std,
                y + y_std,
                alpha=0.2,
                label="±1 std across runs",
            )

            # Mean-only random-prediction baseline overlay
            if not baseline_df.empty and mean_col in baseline_df.columns:
                bdf = baseline_df
                if has_eval_mode and eval_mode is not None:
                    bdf = bdf[bdf["eval_mode"] == eval_mode]
                bdf = bdf.sort_values("threshold")
                _plot_baseline_mean(bdf["threshold"].to_numpy(), bdf[mean_col].to_numpy())

            best_i = int(np.nanargmax(y))

            plt.scatter([x[best_i]], [y[best_i]], s=120)

            plt.axvline(
                x=x[best_i],
                linestyle="--",
                alpha=0.5,
            )

            plt.annotate(
                f"Best λ={x[best_i]:.1f}\n{metric}={y[best_i]:.4f}",
                xy=(x[best_i], y[best_i]),
                xytext=(10, 10),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->"),
            )

            _overlay_crossings(crossings_df)

            title = f"Self Evaluation: {metric.upper()} Mean ± Std\nModel={model_name}"
            if eval_mode is not None:
                title += f", Eval={eval_mode}"
            plt.title(title)

            if para_type == 'volatility':
                plt.xlabel("Label Threshold Multiplier λ")
            elif para_type == 'horizon':    
                plt.xlabel("Label bar horizon h")
            plt.ylabel(metric.upper())

            plt.grid(True, alpha=0.3)

            handles, labels = plt.gca().get_legend_handles_labels()
            handles.append(Line2D([], [], linestyle="none", label=info))
            plt.legend(handles=handles, loc="best", framealpha=0.9)

            plt.tight_layout()

            filename = f"self_eval_{metric}.png" if eval_mode is None else f"self_eval_{metric}_{eval_mode}.png"
            out_path = os.path.join(model_save_dir, filename)

            plt.savefig(out_path, dpi=250)
            plt.close()

            out_paths.append(out_path)

    return out_paths

def plot_financial_return_mean_std(
    financial_summary_df: pd.DataFrame,
    save_dir: str,
    metric: str,
    para_type: str = "volatility",
    crossings_df: pd.DataFrame = None,
) -> list[str]:
    """
    Plot threshold-financial metric mean/std curves.

    Real models:
        - plot mean curve
        - plot ±1 std band

    RandomPredict:
        - used only as baseline
        - overlaid on each real model figure
        - plot mean curve only
        - no std band
        - no separate RandomPredict figure

    crossings_df: optional dataframe from crossing_specificity.load_crossings()
    (columns curve/series/crossing_order/x). When given, each crossing x0 is
    drawn as a vertical dashed line so the crossing points can be visually
    compared against the financial return curve directly.
    """

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    required_cols = [
        "model",
        "fee_rate",
        "threshold",
        mean_col,
        std_col,
        "n_runs",
        "test_size",
    ]

    missing = [c for c in required_cols if c not in financial_summary_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for financial plot: {missing}")

    out_paths = []

    # Split RandomPredict from real models.
    random_df = financial_summary_df[
        financial_summary_df["model"] == BASELINE_MODEL_NAME
    ].copy()

    real_df = financial_summary_df[
        financial_summary_df["model"] != BASELINE_MODEL_NAME
    ].copy()

    for (model_name, fee_rate), df_one in real_df.groupby(
        ["model", "fee_rate"],
        dropna=False,
    ):
        model_save_dir = os.path.join(save_dir, str(model_name), "financial_return")
        os.makedirs(model_save_dir, exist_ok=True)

        df_plot = df_one.sort_values("threshold").copy()

        x = _scale_x(df_plot["threshold"], para_type)
        y = df_plot[mean_col].to_numpy(dtype=float)
        y_std = df_plot[std_col].fillna(0).to_numpy(dtype=float)

        if len(x) == 0:
            continue

        test_size = int(df_plot["test_size"].iloc[0])
        n_runs = int(df_plot["n_runs"].max())

        info = (
            f"{n_runs} independent runs\n"
            f"test samples = {test_size}\n"
            f"fee_rate = {fee_rate:.6f}"
        )

        plt.figure(figsize=(10, 6))

        # =====================================================
        # Real model: mean + std
        # =====================================================
        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=f"{model_name} {metric} mean",
        )

        plt.fill_between(
            x,
            y - y_std,
            y + y_std,
            alpha=0.2,
            label=f"{model_name} ±1 std",
        )

        # =====================================================
        # RandomPredict baseline: mean only, no std
        # =====================================================
        random_one = random_df[
            random_df["fee_rate"] == fee_rate
        ].sort_values("threshold")

        if len(random_one) > 0:
            x_random = _scale_x(random_one["threshold"], para_type)
            y_random = random_one[mean_col].to_numpy(dtype=float)

            plt.plot(
                x_random,
                y_random,
                marker="x",
                linewidth=2,
                linestyle="--",
                label="RandomPredict baseline mean",
            )

        # Mark best point of the real model only.
        if np.any(np.isfinite(y)):
            best_i = int(np.nanargmax(y))

            plt.scatter([x[best_i]], [y[best_i]], s=120)

            plt.axvline(
                x=x[best_i],
                linestyle="--",
                alpha=0.5,
            )

            plt.annotate(
                f"Best λ={x[best_i]:.1f}\n{metric}={y[best_i]:.6f}",
                xy=(x[best_i], y[best_i]),
                xytext=(10, 10),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->"),
            )

        plt.axhline(0.0, linestyle="--", linewidth=1, alpha=0.6)

        _overlay_crossings(crossings_df)

        plt.title(
            f"Financial Diagnostic: {metric}\n"
            f"Model={model_name}, Fee={fee_rate:.6f}"
        )

        if para_type == "volatility":
            plt.xlabel("Label Threshold Multiplier λ")
        elif para_type == "horizon":
            plt.xlabel("Label bar horizon h")

        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)

        handles, _ = plt.gca().get_legend_handles_labels()
        handles.append(Line2D([], [], linestyle="none", label=info))
        plt.legend(handles=handles, loc="best", framealpha=0.9)

        plt.tight_layout()

        out_path = os.path.join(
            model_save_dir,
            f"financial_{metric}_fee_{fee_rate:.6f}.png",
        )

        plt.savefig(out_path, dpi=250)
        plt.close()

        out_paths.append(out_path)

    return out_paths

def plot_financial_three_means(
    financial_summary_df: pd.DataFrame,
    save_dir: str,
    para_type: str = "volatility",
    random_metric: str = "strategy_total_return_mean",
) -> list[str]:
    """
    Financial three-in-one plot.

    For each real model + fee_rate, generate one figure with:
    1) signal_avg_return_mean
    2) strategy_total_return_mean
    3) RandomPredict baseline mean

    random_metric:
        - "strategy_total_return_mean"  -> use RandomPredict strategy mean
        - "signal_avg_return_mean"      -> use RandomPredict signal mean
    """

    required_cols = [
        "model",
        "fee_rate",
        "threshold",
        "n_runs",
        "test_size",
        "signal_avg_return_mean",
        "strategy_total_return_mean",
    ]

    missing = [c for c in required_cols if c not in financial_summary_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for financial three-metric plot: {missing}")

    if random_metric not in ["strategy_total_return_mean", "signal_avg_return_mean"]:
        raise ValueError(
            f"Unsupported random_metric={random_metric}. "
            f"Expected 'strategy_total_return_mean' or 'signal_avg_return_mean'."
        )

    out_paths = []

    random_df = financial_summary_df[
        financial_summary_df["model"] == "RandomPredict"
    ].copy()

    real_df = financial_summary_df[
        financial_summary_df["model"] != "RandomPredict"
    ].copy()

    for (model_name, fee_rate), df_one in real_df.groupby(
        ["model", "fee_rate"],
        dropna=False,
    ):
        model_save_dir = os.path.join(save_dir, str(model_name), "financial_return")
        os.makedirs(model_save_dir, exist_ok=True)

        df_plot = df_one.sort_values("threshold").copy()

        if len(df_plot) == 0:
            continue

        x = _scale_x(df_plot["threshold"], para_type)
        y_signal = df_plot["signal_avg_return_mean"].to_numpy(dtype=float)
        y_strategy = df_plot["strategy_total_return_mean"].to_numpy(dtype=float)

        test_size = int(df_plot["test_size"].iloc[0])
        n_runs = int(df_plot["n_runs"].max())

        info = (
            f"{n_runs} independent runs\n"
            f"test samples = {test_size}\n"
            f"fee_rate = {fee_rate:.6f}"
        )

        plt.figure(figsize=(10, 6))

        # 1) real model signal mean
        plt.plot(
            x,
            y_signal,
            marker="o",
            linewidth=2,
            label="Signal avg return mean",
        )

        # 2) real model strategy mean
        plt.plot(
            x,
            y_strategy,
            marker="s",
            linewidth=2,
            label="Strategy total return mean",
        )

        # 3) random baseline mean
        random_one = random_df[
            random_df["fee_rate"] == fee_rate
        ].sort_values("threshold")

        if len(random_one) > 0 and random_metric in random_one.columns:
            x_random = _scale_x(random_one["threshold"], para_type)
            y_random = random_one[random_metric].to_numpy(dtype=float)

            random_label = (
                "RandomPredict strategy mean"
                if random_metric == "strategy_total_return_mean"
                else "RandomPredict signal mean"
            )

            plt.plot(
                x_random,
                y_random,
                marker="x",
                linewidth=2,
                linestyle="--",
                label=random_label,
            )

        plt.axhline(0.0, linestyle="--", linewidth=1, alpha=0.6)

        plt.title(
            f"Financial Summary: Signal / Strategy / Random\n"
            f"Model={model_name}, Fee={fee_rate:.6f}"
        )

        if para_type == "volatility":
            plt.xlabel("Label Threshold Multiplier λ")
        elif para_type == "horizon":
            plt.xlabel("Label bar horizon h")
        else:
            plt.xlabel("Label parameter")

        plt.ylabel("Return")
        plt.grid(True, alpha=0.3)

        handles, _ = plt.gca().get_legend_handles_labels()
        handles.append(Line2D([], [], linestyle="none", label=info))
        plt.legend(handles=handles, loc="best", framealpha=0.9)

        plt.tight_layout()

        out_path = os.path.join(
            model_save_dir,
            f"financial_three_means_fee_{fee_rate:.6f}.png",
        )

        plt.savefig(out_path, dpi=250)
        plt.close()

        out_paths.append(out_path)

    return out_paths

def plot_cross_train_batch_summary(
    cross_train_batch_summary_df,
    save_dir,
    metric: str,
    total_runs,
    para_type:str = "volatility",
    baseline_df: pd.DataFrame = None,
):
    """
    baseline_df: optional self-eval-shaped summary (columns: model, eval_mode,
    threshold, "{metric}_mean") already filtered/derived from
    self_summary_df's RandomPredict rows. Since a random predictor's
    performance does not depend on which threshold it was "trained" at, the
    baseline is drawn as a flat horizontal line at the average of its
    self-eval mean across all thresholds, for the matching eval_mode.
    """
    mean_col = f"{metric}_mean_mean"
    std_col = f"{metric}_mean_std"
    baseline_self_col = f"{metric}_mean"

    for (model, eval_mode), df_one in cross_train_batch_summary_df.groupby(
        ["model", "eval_mode"]
    ):
        model_save_dir = os.path.join(save_dir, model)
        os.makedirs(model_save_dir, exist_ok=True)

        df_plot = df_one.sort_values("train_threshold").copy()

        x = _scale_x(df_plot["train_threshold"], para_type)
        y = df_plot[mean_col].to_numpy()
        y_std = df_plot[std_col].fillna(0).to_numpy()

        if len(x) == 0:
            continue

        train_size = int(df_plot["train_size"].iloc[0])
        test_size = int(df_plot["test_size"].iloc[0])

        info = (
            f"{total_runs} independent runs\n"
            f"train samples = {train_size}\n"
            f"test samples = {test_size}"
        )

        plt.figure(figsize=(10, 6))

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=f"{metric} mean",
        )

        plt.fill_between(
            x,
            y - y_std,
            y + y_std,
            alpha=0.2,
            label="±1 std across runs",
        )

        # Mean-only random-prediction baseline: flat line (train_threshold-agnostic)
        if baseline_df is not None and not baseline_df.empty and baseline_self_col in baseline_df.columns:
            bdf = baseline_df[baseline_df["eval_mode"] == eval_mode] if "eval_mode" in baseline_df.columns else baseline_df
            if not bdf.empty:
                baseline_value = float(bdf[baseline_self_col].mean())
                _plot_baseline_mean(x, np.full_like(x, baseline_value, dtype=float))

        best_i = int(np.nanargmax(y))

        plt.scatter([x[best_i]], [y[best_i]], s=120)

        plt.annotate(
            f"Best λ={x[best_i]:.1f}\n{metric}={y[best_i]:.4f}",
            xy=(x[best_i], y[best_i]),
            xytext=(10, 10),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->"),
        )

        plt.title(
            f"Train Threshold Robustness\n"
            f"Model={model}, Eval={eval_mode}"
        )

        if para_type == 'volatility':
            plt.xlabel("Label Train Threshold Multiplier λ")
        elif para_type == 'horizon':    
            plt.xlabel("Label Train bar horizon h")
        plt.ylabel(metric.upper())

        plt.grid(True, alpha=0.3)

        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(Line2D([], [], linestyle="none", label=info))
        plt.legend(handles=handles, loc="best", framealpha=0.9)

        plt.tight_layout()

        out_path = os.path.join(
            model_save_dir,
            f"cross_train_batch_{metric}_{model}_{eval_mode}.png",
        )

        plt.savefig(out_path, dpi=250)
        plt.close()

def plot_cross_eval_batch_summary(
    cross_eval_batch_summary_df,
    save_dir,
    metric: str,
    total_runs,
    para_type:str = "volatility",
    baseline_df: pd.DataFrame = None,
):
    """
    baseline_df: optional self-eval-shaped summary (columns: model, eval_mode,
    threshold, "{metric}_mean") derived from self_summary_df's RandomPredict
    rows. Evaluating a random predictor on eval regime X is numerically the
    same computation as the self-eval baseline for threshold X, so the curve
    is drawn directly against the same threshold/eval_threshold grid.
    """
    mean_col = f"{metric}_mean_mean"
    std_col = f"{metric}_mean_std"
    baseline_self_col = f"{metric}_mean"

    for (model, eval_mode), df_one in cross_eval_batch_summary_df.groupby(
        ["model", "eval_mode"]
    ):
        model_save_dir = os.path.join(save_dir, model)
        os.makedirs(model_save_dir, exist_ok=True)

        df_plot = df_one.sort_values("eval_threshold").copy()

        x = _scale_x(df_plot["eval_threshold"], para_type)
        y = df_plot[mean_col].to_numpy()
        y_std = df_plot[std_col].fillna(0).to_numpy()

        if len(x) == 0:
            continue

        train_size = int(df_plot["train_size"].iloc[0])
        test_size = int(df_plot["test_size"].iloc[0])

        info = (
            f"{total_runs} independent runs\n"
            f"train samples = {train_size}\n"
            f"test samples = {test_size}"
        )

        plt.figure(figsize=(10, 6))

        plt.plot(
            x,
            y,
            marker="s",
            linewidth=2,
            linestyle="--",
            label=f"{metric} mean",
        )

        plt.fill_between(
            x,
            y - y_std,
            y + y_std,
            alpha=0.2,
            label="±1 std",
        )

        # Mean-only random-prediction baseline, aligned on the same threshold grid
        if baseline_df is not None and not baseline_df.empty and baseline_self_col in baseline_df.columns:
            bdf = baseline_df[baseline_df["eval_mode"] == eval_mode] if "eval_mode" in baseline_df.columns else baseline_df
            if not bdf.empty:
                bdf = bdf.sort_values("threshold")
                _plot_baseline_mean(_scale_x(bdf["threshold"], para_type), bdf[baseline_self_col].to_numpy())

        easiest_i = int(np.nanargmax(y))

        plt.scatter([x[easiest_i]], [y[easiest_i]], s=120)

        plt.annotate(
            f"Easiest λ={x[easiest_i]:.1f}\n{metric}={y[easiest_i]:.4f}",
            xy=(x[easiest_i], y[easiest_i]),
            xytext=(10, -15),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->"),
        )

        plt.title(
            f"Eval Threshold Separability\n"
            f"Model={model}, Eval={eval_mode}"
        )

        if para_type == 'volatility':
            plt.xlabel("Label Eval Threshold Multiplier λ")
        elif para_type == 'horizon':    
            plt.xlabel("Label Eval bar horizon h")
        plt.ylabel(metric.upper())

        plt.grid(True, alpha=0.3)

        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(Line2D([], [], linestyle="none", label=info))
        plt.legend(handles=handles, loc="best", framealpha=0.9)

        plt.tight_layout()

        out_path = os.path.join(
            model_save_dir,
            f"cross_eval_batch_{metric}_{model}_{eval_mode}.png",
        )

        plt.savefig(out_path, dpi=250)
        plt.close()

def plot_train_vs_eval_summary(cross_train_batch_summary_df, cross_eval_batch_summary_df, save_dir,
                                metric: str, total_runs, para_type: str = "volatility",
                                baseline_df: pd.DataFrame = None):
    mean_col = f"{metric}_mean_mean"
    baseline_self_col = f"{metric}_mean"

    if para_type == 'volatility':
        label = "Threshold"
    elif para_type == 'horizon':
        label = "Bar horizon"
    else:
        label = "Threshold"

    for model in sorted(cross_train_batch_summary_df["model"].unique()):
        model_save_dir = os.path.join(save_dir, model)
        os.makedirs(model_save_dir, exist_ok=True)

        for eval_mode in sorted(cross_train_batch_summary_df["eval_mode"].unique()):
            train_df = cross_train_batch_summary_df.query(
                "model == @model and eval_mode == @eval_mode"
            ).sort_values("train_threshold")

            eval_df = cross_eval_batch_summary_df.query(
                "model == @model and eval_mode == @eval_mode"
            ).sort_values("eval_threshold")

            if train_df.empty or eval_df.empty:
                continue

            train_size = int(train_df["train_size"].iloc[0])
            test_size = int(eval_df["test_size"].iloc[0])
            info = (
                f"{total_runs} independent runs\n"
                f"train samples = {train_size}\n"
                f"test samples = {test_size}"
            )
            fig, ax = plt.subplots(figsize=(12, 7))

            ax.plot(
                _scale_x(train_df["train_threshold"], para_type),
                train_df[mean_col],
                marker="o",
                linewidth=3,
                label=f"Train {label} Robustness",
            )
            ax.plot(
                _scale_x(eval_df["eval_threshold"], para_type),
                eval_df[mean_col],
                marker="s",
                linewidth=3,
                linestyle="--",
                label=f"Eval {label} Separability",
            )

            # Mean-only random-prediction baseline: flat line for "train" axis
            # (random doesn't depend on train_threshold) + curve for "eval" axis
            if baseline_df is not None and not baseline_df.empty and baseline_self_col in baseline_df.columns:
                bdf = baseline_df[baseline_df["eval_mode"] == eval_mode] if "eval_mode" in baseline_df.columns else baseline_df
                if not bdf.empty:
                    bdf = bdf.sort_values("threshold")
                    baseline_value = float(bdf[baseline_self_col].mean())
                    x_train = _scale_x(train_df["train_threshold"], para_type)
                    _plot_baseline_mean(
                        x_train, np.full_like(x_train, baseline_value, dtype=float),
                        label="Baseline (Random) Train",
                    )
                    _plot_baseline_mean(
                        _scale_x(bdf["threshold"], para_type), bdf[baseline_self_col].to_numpy(),
                        label="Baseline (Random) Eval",
                    )

            handles, _ = ax.get_legend_handles_labels()
            handles.append(Line2D([], [], linestyle="none", label=info))

            ax.legend(handles=handles, loc="upper right", framealpha=0.9)
            ax.set(
                title=f"Train vs Eval Threshold Analysis\nModel={model}, Eval={eval_mode}",
                xlabel="Threshold λ",
                ylabel=metric.upper(),
            )
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(
                os.path.join(model_save_dir, f"train_vs_eval_{metric}_{model}_{eval_mode}.png"),
                dpi=250,
            )
            plt.close(fig)

def plot_self_cross_combined_summary(
    self_summary_df: pd.DataFrame,
    cross_train_batch_summary_df: pd.DataFrame,
    cross_eval_batch_summary_df: pd.DataFrame,
    save_dir: str,
    metric: str,
    total_runs: int,
    para_type: str = "volatility",
    crossings_df: pd.DataFrame = None,
) -> list[str]:
    """
    Overlay Self-Eval / Cross-Train / Cross-Eval mean curves (no std) on one figure,
    one figure per (model, eval_mode). metric in {'accuracy', 'macro_f1', 'mcc'}.

    Produces, per model, 3 (metric) * 2 (eval_mode) = 6 files named
    "{metric}_{eval_mode}.png", e.g. accuracy_balanced.png, accuracy_raw.png,
    macro_f1_balanced.png, macro_f1_raw.png, mcc_balanced.png, mcc_raw.png.

    Rows with model == BASELINE_MODEL_NAME ("RandomPredict") in
    self_summary_df are not plotted as their own model; instead a single
    "Baseline (Random)" mean-only curve is overlaid on every real model's
    figure (self-eval value == cross-eval value on the same threshold grid,
    since a random predictor's score only depends on which regime it is
    evaluated against, not on training).
    """

    metric_col_map = {
        "accuracy": ("accuracy_mean", "accuracy_mean_mean"),
        "macro_f1": ("macro_f1_mean", "macro_f1_mean_mean"),
        "mcc": ("mcc_mean", "mcc_mean_mean"),
    }
    if metric not in metric_col_map:
        raise ValueError(f"Unsupported metric '{metric}', expected one of {list(metric_col_map)}")

    self_col, cross_col = metric_col_map[metric]

    out_paths = []
    xlabel = "Label Threshold Multiplier λ" if para_type == "volatility" else "Label bar horizon h"

    real_self_df, baseline_self_df = _split_baseline(self_summary_df)

    for model_name in sorted(real_self_df["model"].unique()):
        model_save_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)

        for eval_mode in ["balanced", "raw"]:
            self_df = real_self_df.query(
                "model == @model_name and eval_mode == @eval_mode"
            ).sort_values("threshold")
            train_df = cross_train_batch_summary_df.query(
                "model == @model_name and eval_mode == @eval_mode"
            ).sort_values("train_threshold")
            eval_df = cross_eval_batch_summary_df.query(
                "model == @model_name and eval_mode == @eval_mode"
            ).sort_values("eval_threshold")

            if self_df.empty and train_df.empty and eval_df.empty:
                continue

            train_size = int(self_df["train_size"].iloc[0]) if not self_df.empty else (
                int(train_df["train_size"].iloc[0]) if not train_df.empty else None
            )
            test_size = int(self_df["test_size"].iloc[0]) if not self_df.empty else (
                int(eval_df["test_size"].iloc[0]) if not eval_df.empty else None
            )

            info_lines = [f"{total_runs} independent runs"]
            if train_size is not None:
                info_lines.append(f"train samples = {train_size}")
            if test_size is not None:
                info_lines.append(f"test samples = {test_size}")
            info = "\n".join(info_lines)

            plt.figure(figsize=(10, 6))

            if not self_df.empty:
                plt.plot(
                    _scale_x(self_df["threshold"], para_type), self_df[self_col],
                    marker="o", linewidth=2, label="Self Eval",
                )
            if not train_df.empty:
                plt.plot(
                    _scale_x(train_df["train_threshold"], para_type), train_df[cross_col],
                    marker="s", linestyle="--", linewidth=2, label="Cross Train",
                )
            if not eval_df.empty:
                plt.plot(
                    _scale_x(eval_df["eval_threshold"], para_type), eval_df[cross_col],
                    marker="^", linestyle=":", linewidth=2, label="Cross Eval",
                )

            # Mean-only random-prediction baseline (single representative curve)
            if not baseline_self_df.empty and self_col in baseline_self_df.columns:
                bdf = baseline_self_df[baseline_self_df["eval_mode"] == eval_mode].sort_values("threshold")
                if not bdf.empty:
                    _plot_baseline_mean(_scale_x(bdf["threshold"], para_type), bdf[self_col].to_numpy())

            _overlay_crossings(crossings_df)

            plt.title(
                f"{metric.upper()} Mean: Self vs Cross-Train vs Cross-Eval\n"
                f"Model={model_name}, Eval={eval_mode}"
            )
            plt.xlabel(xlabel)
            plt.ylabel(metric.upper())
            plt.grid(True, alpha=0.3)

            handles, _ = plt.gca().get_legend_handles_labels()
            handles.append(Line2D([], [], linestyle="none", label=info))
            plt.legend(handles=handles, loc="best", framealpha=0.9)

            plt.tight_layout()

            out_path = os.path.join(model_save_dir, f"{metric}_{eval_mode}.png")
            plt.savefig(out_path, dpi=250)
            plt.close()

            out_paths.append(out_path)

    return out_paths

def plot_self_cross_combined_class_summary(
    self_summary_df: pd.DataFrame,
    cross_train_batch_summary_df: pd.DataFrame,
    cross_eval_batch_summary_df: pd.DataFrame,
    save_dir: str,
    cls: str,
    metric: str,
    total_runs: int,
    para_type: str = "volatility",
    crossings_df: pd.DataFrame = None,
) -> list[str]:
    """
    Same idea as plot_self_cross_combined_summary, but for the per-class
    (positive / negative) precision / recall / f1 metrics.

    cls in {'positive', 'negative'}; metric in {'precision', 'recall', 'f1'}.

    Called for 2 (class) * 3 (metric) combinations, each internally looping
    over eval_mode in {'balanced', 'raw'}, producing 2*3*2 = 12 files per
    model named "{class}_{metric}_{eval_mode}.png", e.g.
    positive_precision_balanced.png, negative_recall_raw.png, ...
    """
    class_suffix_map = {"positive": "pos", "negative": "neg"}
    metric_prefix_map = {"precision": "p", "recall": "r", "f1": "f"}

    if cls not in class_suffix_map:
        raise ValueError(f"Unsupported cls '{cls}', expected one of {list(class_suffix_map)}")
    if metric not in metric_prefix_map:
        raise ValueError(f"Unsupported metric '{metric}', expected one of {list(metric_prefix_map)}")

    suffix = class_suffix_map[cls]
    prefix = metric_prefix_map[metric]

    self_col = f"{prefix}_{suffix}_mean"
    cross_col = f"{prefix}_{suffix}_mean_mean"

    out_paths = []
    xlabel = "Label Threshold Multiplier λ" if para_type == "volatility" else "Label bar horizon h"

    real_self_df, baseline_self_df = _split_baseline(self_summary_df)

    for model_name in sorted(real_self_df["model"].unique()):
        model_save_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)

        for eval_mode in ["balanced", "raw"]:
            self_df = real_self_df.query(
                "model == @model_name and eval_mode == @eval_mode"
            ).sort_values("threshold")
            train_df = cross_train_batch_summary_df.query(
                "model == @model_name and eval_mode == @eval_mode"
            ).sort_values("train_threshold")
            eval_df = cross_eval_batch_summary_df.query(
                "model == @model_name and eval_mode == @eval_mode"
            ).sort_values("eval_threshold")

            if self_df.empty and train_df.empty and eval_df.empty:
                continue

            train_size = int(self_df["train_size"].iloc[0]) if not self_df.empty else (
                int(train_df["train_size"].iloc[0]) if not train_df.empty else None
            )
            test_size = int(self_df["test_size"].iloc[0]) if not self_df.empty else (
                int(eval_df["test_size"].iloc[0]) if not eval_df.empty else None
            )

            info_lines = [f"{total_runs} independent runs"]
            if train_size is not None:
                info_lines.append(f"train samples = {train_size}")
            if test_size is not None:
                info_lines.append(f"test samples = {test_size}")
            info = "\n".join(info_lines)

            plt.figure(figsize=(10, 6))

            if not self_df.empty and self_col in self_df.columns:
                plt.plot(
                    _scale_x(self_df["threshold"], para_type), self_df[self_col],
                    marker="o", linewidth=2, label="Self Eval",
                )
            if not train_df.empty and cross_col in train_df.columns:
                plt.plot(
                    _scale_x(train_df["train_threshold"], para_type), train_df[cross_col],
                    marker="s", linestyle="--", linewidth=2, label="Cross Train",
                )
            if not eval_df.empty and cross_col in eval_df.columns:
                plt.plot(
                    _scale_x(eval_df["eval_threshold"], para_type), eval_df[cross_col],
                    marker="^", linestyle=":", linewidth=2, label="Cross Eval",
                )

            # Mean-only random-prediction baseline (single representative curve)
            if not baseline_self_df.empty and self_col in baseline_self_df.columns:
                bdf = baseline_self_df[baseline_self_df["eval_mode"] == eval_mode].sort_values("threshold")
                if not bdf.empty:
                    _plot_baseline_mean(_scale_x(bdf["threshold"], para_type), bdf[self_col].to_numpy())

            _overlay_crossings(crossings_df)

            plt.title(
                f"{cls.capitalize()} class {metric}: Self vs Cross-Train vs Cross-Eval\n"
                f"Model={model_name}, Eval={eval_mode}"
            )
            plt.xlabel(xlabel)
            plt.ylabel(metric.capitalize())
            plt.grid(True, alpha=0.3)

            handles, _ = plt.gca().get_legend_handles_labels()
            handles.append(Line2D([], [], linestyle="none", label=info))
            plt.legend(handles=handles, loc="best", framealpha=0.9)

            plt.tight_layout()

            out_path = os.path.join(model_save_dir, f"{cls}_{metric}_{eval_mode}.png")
            plt.savefig(out_path, dpi=250)
            plt.close()

            out_paths.append(out_path)

    return out_paths

def plot_self_eval_three_metrics(
    self_summary_df: pd.DataFrame,
    save_dir: str,
    para_type: str = "volatility",
) -> list[str]:
    """
    Self evaluation three-metric plot:
    accuracy / macro_f1 / mcc in one figure.

    Use mean curves only.
    Existing single-metric mean±std figures are kept unchanged.
    """

    metrics = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("mcc", "MCC"),
    ]

    out_paths = []

    for model_name, df_one in self_summary_df.groupby("model", dropna=False):
        model_save_dir = os.path.join(save_dir, str(model_name))
        os.makedirs(model_save_dir, exist_ok=True)

        df_plot = df_one.sort_values("threshold").copy()

        if len(df_plot) == 0:
            continue

        x = _scale_x(df_plot["threshold"], para_type)

        plt.figure(figsize=(10, 6))

        for metric, label in metrics:
            mean_col = f"{metric}_mean"

            if mean_col not in df_plot.columns:
                continue

            y = df_plot[mean_col].to_numpy(dtype=float)

            plt.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                label=label,
            )

        if para_type == "volatility":
            xlabel = "Label Threshold Multiplier λ"
        elif para_type == "horizon":
            xlabel = "Label bar horizon h"
        else:
            xlabel = "Label parameter"

        train_size = int(df_plot["train_size"].iloc[0])
        test_size = int(df_plot["test_size"].iloc[0])
        n_runs = int(df_plot["n_runs"].max())

        info = (
            f"{n_runs} independent runs\n"
            f"train samples = {train_size}\n"
            f"test samples = {test_size}"
        )

        plt.title(
            f"Self Evaluation: Accuracy / Macro-F1 / MCC\n"
            f"Model={model_name}"
        )
        plt.xlabel(xlabel)
        plt.ylabel("Metric value")
        plt.grid(True, alpha=0.3)

        handles, _ = plt.gca().get_legend_handles_labels()
        handles.append(Line2D([], [], linestyle="none", label=info))
        plt.legend(handles=handles, loc="best", framealpha=0.9)

        plt.tight_layout()

        out_path = os.path.join(
            model_save_dir,
            "self_eval_three_metrics.png",
        )

        plt.savefig(out_path, dpi=250)
        plt.close()

        out_paths.append(out_path)

    return out_paths

def plot_cross_train_three_metrics(
    cross_train_batch_summary_df: pd.DataFrame,
    save_dir: str,
    total_runs: int,
    para_type: str = "volatility",
) -> list[str]:
    """
    Cross-train three-metric plot:
    accuracy / macro_f1 / mcc in one figure.

    One figure for each model + eval_mode.
    eval_mode naturally distinguishes raw / balanced.
    """

    metrics = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("mcc", "MCC"),
    ]

    out_paths = []

    for (model, eval_mode), df_one in cross_train_batch_summary_df.groupby(
        ["model", "eval_mode"],
        dropna=False,
    ):
        model_save_dir = os.path.join(save_dir, str(model))
        os.makedirs(model_save_dir, exist_ok=True)

        df_plot = df_one.sort_values("train_threshold").copy()

        if len(df_plot) == 0:
            continue

        x = _scale_x(df_plot["train_threshold"], para_type)

        plt.figure(figsize=(10, 6))

        for metric, label in metrics:
            mean_col = f"{metric}_mean_mean"

            if mean_col not in df_plot.columns:
                continue

            y = df_plot[mean_col].to_numpy(dtype=float)

            plt.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                label=label,
            )

        if para_type == "volatility":
            xlabel = "Label Train Threshold Multiplier λ"
        elif para_type == "horizon":
            xlabel = "Label Train bar horizon h"
        else:
            xlabel = "Train label parameter"

        train_size = int(df_plot["train_size"].iloc[0])
        test_size = int(df_plot["test_size"].iloc[0])

        info = (
            f"{total_runs} independent runs\n"
            f"train samples = {train_size}\n"
            f"test samples = {test_size}"
        )

        plt.title(
            f"Cross Train Summary: Accuracy / Macro-F1 / MCC\n"
            f"Model={model}, Eval={eval_mode}"
        )
        plt.xlabel(xlabel)
        plt.ylabel("Metric value")
        plt.grid(True, alpha=0.3)

        handles, _ = plt.gca().get_legend_handles_labels()
        handles.append(Line2D([], [], linestyle="none", label=info))
        plt.legend(handles=handles, loc="best", framealpha=0.9)

        plt.tight_layout()

        out_path = os.path.join(
            model_save_dir,
            f"cross_train_three_metrics_{eval_mode}.png",
        )

        plt.savefig(out_path, dpi=250)
        plt.close()

        out_paths.append(out_path)

    return out_paths

def plot_cross_eval_three_metrics(
    cross_eval_batch_summary_df: pd.DataFrame,
    save_dir: str,
    total_runs: int,
    para_type: str = "volatility",
) -> list[str]:
    """
    Cross-eval three-metric plot:
    accuracy / macro_f1 / mcc in one figure.

    One figure for each model + eval_mode.
    eval_mode naturally distinguishes raw / balanced.
    """

    metrics = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("mcc", "MCC"),
    ]

    out_paths = []

    for (model, eval_mode), df_one in cross_eval_batch_summary_df.groupby(
        ["model", "eval_mode"],
        dropna=False,
    ):
        model_save_dir = os.path.join(save_dir, str(model))
        os.makedirs(model_save_dir, exist_ok=True)

        df_plot = df_one.sort_values("eval_threshold").copy()

        if len(df_plot) == 0:
            continue

        x = _scale_x(df_plot["eval_threshold"], para_type)

        plt.figure(figsize=(10, 6))

        for metric, label in metrics:
            mean_col = f"{metric}_mean_mean"

            if mean_col not in df_plot.columns:
                continue

            y = df_plot[mean_col].to_numpy(dtype=float)

            plt.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                label=label,
            )

        if para_type == "volatility":
            xlabel = "Label Eval Threshold Multiplier λ"
        elif para_type == "horizon":
            xlabel = "Label Eval bar horizon h"
        else:
            xlabel = "Eval label parameter"

        train_size = int(df_plot["train_size"].iloc[0])
        test_size = int(df_plot["test_size"].iloc[0])

        info = (
            f"{total_runs} independent runs\n"
            f"train samples = {train_size}\n"
            f"test samples = {test_size}"
        )

        plt.title(
            f"Cross Eval Summary: Accuracy / Macro-F1 / MCC\n"
            f"Model={model}, Eval={eval_mode}"
        )
        plt.xlabel(xlabel)
        plt.ylabel("Metric value")
        plt.grid(True, alpha=0.3)

        handles, _ = plt.gca().get_legend_handles_labels()
        handles.append(Line2D([], [], linestyle="none", label=info))
        plt.legend(handles=handles, loc="best", framealpha=0.9)

        plt.tight_layout()

        out_path = os.path.join(
            model_save_dir,
            f"cross_eval_three_metrics_{eval_mode}.png",
        )

        plt.savefig(out_path, dpi=250)
        plt.close()

        out_paths.append(out_path)

    return out_paths

def summarize_cross_row_mean(
    cross_eval_df: pd.DataFrame,
    metric: str = "macro_f1",
) -> pd.DataFrame:
    row_by_run = (
        cross_eval_df
        .groupby(["model", "eval_mode", "run_id", "train_threshold"])[metric]
        .mean()
        .reset_index(name=f"row_mean_{metric}")
    )

    row_summary = (
        row_by_run
        .groupby(["model", "eval_mode", "train_threshold"])
        .agg(
            row_mean=(f"row_mean_{metric}", "mean"),
            row_std=(f"row_mean_{metric}", "std"),
            n_runs=("run_id", "nunique"),
        )
        .reset_index()
        .sort_values(["model", "eval_mode", "train_threshold"])
    )
    return row_summary
# -----------------------------
# Orchestration: run every plot function against a set of summary tables
# -----------------------------
def generate_all_plots(
    save_dir: str,
    para_type: str,
    total_runs: int,
    self_summary_df: pd.DataFrame,
    cross_train_batch_summary_df: pd.DataFrame,
    cross_eval_batch_summary_df: pd.DataFrame,
    cross_matrix_summary_df: pd.DataFrame,
    financial_summary_df: pd.DataFrame,
    crossings_df: pd.DataFrame = None,
) -> None:
    """
    Generate every figure for one experiment output directory.

    This is a pure function of the summary tables (+ a couple of scalars for
    titles/labels) — it has no knowledge of models, training, or raw
    per-window data. It is called:
      - automatically by train.py right after it saves the summary CSVs
      - standalone by `python plot_results.py --save_dir ...`, via load_and_plot()

    self_summary_df is expected to contain rows for model ==
    BASELINE_MODEL_NAME ("RandomPredict"); those are pulled out here once
    and passed down as `baseline_df` so every plotting function overlays a
    single, consistent, mean-only random-prediction baseline instead of
    getting its own model folder/figures.

    crossings_df: optional dataframe from crossing_specificity.load_crossings()
    (columns curve/series/crossing_order/x, on the TRUE parameter scale).
    When given, the self-eval and financial mean/std figures additionally
    draw a vertical line at each crossing x0, so the label-distribution
    crossing points can be visually compared against the performance curves.
    """
    _, baseline_df = _split_baseline(self_summary_df)

    # plot_self_eval_mean_std(self_summary_df, save_dir, metric="macro_f1", para_type=para_type, crossings_df=crossings_df)
    # plot_self_eval_mean_std(self_summary_df, save_dir, metric="accuracy", para_type=para_type, crossings_df=crossings_df)
    # plot_self_eval_mean_std(self_summary_df, save_dir, metric="mcc", para_type=para_type, crossings_df=crossings_df)

    # plot_cross_train_batch_summary(cross_train_batch_summary_df, save_dir, metric="macro_f1", total_runs=total_runs, para_type=para_type, baseline_df=baseline_df)
    # plot_cross_train_batch_summary(cross_train_batch_summary_df, save_dir, metric="mcc", total_runs=total_runs, para_type=para_type, baseline_df=baseline_df)
    # plot_cross_train_batch_summary(cross_train_batch_summary_df, save_dir, metric="accuracy", total_runs=total_runs, para_type=para_type, baseline_df=baseline_df)

    # plot_cross_eval_batch_summary(cross_eval_batch_summary_df, save_dir, metric="macro_f1", total_runs=total_runs, para_type=para_type, baseline_df=baseline_df)
    # plot_cross_eval_batch_summary(cross_eval_batch_summary_df, save_dir, metric="mcc", total_runs=total_runs, para_type=para_type, baseline_df=baseline_df)
    # plot_cross_eval_batch_summary(cross_eval_batch_summary_df, save_dir, metric="accuracy", total_runs=total_runs, para_type=para_type, baseline_df=baseline_df)

    # plot_train_vs_eval_summary(cross_train_batch_summary_df, cross_eval_batch_summary_df, save_dir, metric="macro_f1", total_runs=total_runs, para_type=para_type, baseline_df=baseline_df)
    # plot_train_vs_eval_summary(cross_train_batch_summary_df, cross_eval_batch_summary_df, save_dir, metric="mcc", total_runs=total_runs, para_type=para_type, baseline_df=baseline_df)
    # plot_train_vs_eval_summary(cross_train_batch_summary_df, cross_eval_batch_summary_df, save_dir, metric="accuracy", total_runs=total_runs, para_type=para_type, baseline_df=baseline_df)

    # Self vs Cross-Train vs Cross-Eval, overall metrics (mean only): 3 metrics * 2 eval_modes = 6 files/model
    plot_self_cross_combined_summary(self_summary_df, cross_train_batch_summary_df, cross_eval_batch_summary_df, save_dir, metric="accuracy", total_runs=total_runs, para_type=para_type, crossings_df=crossings_df)
    plot_self_cross_combined_summary(self_summary_df, cross_train_batch_summary_df, cross_eval_batch_summary_df, save_dir, metric="macro_f1", total_runs=total_runs, para_type=para_type, crossings_df=crossings_df)
    plot_self_cross_combined_summary(self_summary_df, cross_train_batch_summary_df, cross_eval_batch_summary_df, save_dir, metric="mcc", total_runs=total_runs, para_type=para_type, crossings_df=crossings_df)

    # Self vs Cross-Train vs Cross-Eval, per-class (positive/negative) metrics (mean only):
    # 2 classes * 3 metrics * 2 eval_modes = 12 files/model
    for cls in ["positive", "negative"]:
        for metric in ["precision", "recall", "f1"]:
            plot_self_cross_combined_class_summary(
                self_summary_df, cross_train_batch_summary_df, cross_eval_batch_summary_df,
                save_dir, cls=cls, metric=metric, total_runs=total_runs, para_type=para_type,
                crossings_df=crossings_df,
            )

    # NOTE: cross_matrix_summary_df (heatmap) has no natural random-baseline
    # overlay — it's a 2D (train_threshold x eval_threshold) matrix per
    # model, not a line plot, so a "mean-only baseline line" doesn't apply.
    # plot_cross_eval_heatmap(cross_matrix_summary_df, save_dir, metric="macro_f1", annotate_std=True, para_type=para_type)
    # plot_cross_eval_heatmap(cross_matrix_summary_df, save_dir, metric="mcc", annotate_std=True, para_type=para_type)
    # plot_cross_eval_heatmap(cross_matrix_summary_df, save_dir, metric="accuracy", annotate_std=True, para_type=para_type)

    plot_financial_return_mean_std(financial_summary_df, save_dir, metric="strategy_total_return", para_type=para_type, crossings_df=crossings_df)
    plot_financial_return_mean_std(financial_summary_df, save_dir, metric="signal_avg_return", para_type=para_type, crossings_df=crossings_df)


# -----------------------------
# Standalone loading: read persisted CSV/JSON from disk, no training required
# -----------------------------

# Maps the kwarg name expected by generate_all_plots -> the CSV filename saved by train.py
_SUMMARY_FILES = {
    "self_summary_df": "self_eval_summary_mean_std.csv",
    "cross_train_batch_summary_df": "cross_train_batch_summary_mean_std.csv",
    "cross_eval_batch_summary_df": "cross_eval_batch_summary_mean_std.csv",
    "cross_matrix_summary_df": "cross_matrix_summary_mean_std.csv",
    "financial_summary_df": "financial_return_summary_mean_std.csv",
}


def load_meta(save_dir: str) -> dict:
    """Read the small meta.json (para_type, total_runs, ...) saved by train.py."""
    meta_path = os.path.join(save_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r") as f:
        return json.load(f)


def load_summary_tables(save_dir: str) -> dict:
    """Read every summary CSV that generate_all_plots needs, from save_dir."""
    tables = {}
    missing = []
    for kwarg_name, filename in _SUMMARY_FILES.items():
        path = os.path.join(save_dir, filename)
        if not os.path.exists(path):
            missing.append(filename)
            continue
        tables[kwarg_name] = pd.read_csv(path)
    if missing:
        raise FileNotFoundError(
            f"Missing summary file(s) under '{save_dir}': {missing}. "
            f"Make sure train.py finished and saved its outputs there first."
        )
    return tables


def load_and_plot(save_dir: str, pre_para: common.BaseDefine) -> None:
    """
    Standalone entrypoint: read persisted summary CSVs (+ meta.json) and
    regenerate every figure, without touching model training/evaluation.

    `save_dir` is the same top-level output dir train.py's `main()` takes
    (e.g. common.TRAIN_OUT_DIR); the actual per-experiment directory is
    derived from it + `pre_para`, mirroring exactly what
    run_fixed_neutral_subsampling_experiment() does in train.py:

        save_dir/{symbol}_{interval}/{label_type}/{para_type}

    total_runs is read from that directory's meta.json (written by train.py)
    instead of being passed in.

    Also tries to load the label-ratio crossing points via
    crossing_specificity.load_crossings() (reads
    output/regime_discovery_output/.../*_crossings.csv, written by
    `preparation.py --mode plot`) so the self-eval/financial figures can
    overlay them. If that file doesn't exist yet (plot mode hasn't been run
    for this para), the figures are still produced, just without the
    crossing overlay.
    """
    exp_dir = os.path.join(
        save_dir, f"{pre_para.symbol}_{pre_para.interval}",
        pre_para.label_type, pre_para.para_type,
    )

    meta = load_meta(exp_dir)
    total_runs = meta.get("total_runs", 0)

    tables = load_summary_tables(exp_dir)

    try:
        crossings_df = crossing_specificity.load_crossings(pre_para)
    except FileNotFoundError as e:
        print(f"[skip] no crossings found for {pre_para.symbol}_{pre_para.interval}/"
              f"{pre_para.label_type}/{pre_para.para_type}, plotting without crossing overlay: {e}")
        crossings_df = None

    generate_all_plots(
        save_dir=exp_dir,
        para_type=pre_para.para_type,
        total_runs=total_runs,
        crossings_df=crossings_df,
        **tables,
    )
    return exp_dir


if __name__ == "__main__":
    for pre_para in [
        # common.BTC_15m_fthl_volatility,
        # common.BTC_15m_fthl_horizon,
        # common.BTC_15m_tbm_volatility,
        # common.BTC_15m_tbm_horizon,
        ]:
        exp_dir = load_and_plot(common.TRAIN_OUT_DIR, pre_para)
        print(f"All plots regenerated under: {exp_dir}")