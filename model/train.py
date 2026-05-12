#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed Neutral Subsampling Experiment (Probe)
- Goal: compare how different label strictness (label_vxx) changes sample composition and affects training
- Design: fix neutral (consensus neutral); for each threshold, resample pos/neg and train an model probe
- Key optimizations:
  1) Remove StandardScaler (dataset already does in-window normalization)
  2) Flatten X only once; cache flattened test set; avoid repeated reshape inside the loop
  3) No GPU preload (sklearn runs on CPU)
  4) Remove unrelated model configs / training pipeline / samplers, etc.
"""

import os,sys
import time
import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
from data_process import common
from model.data_loader import TimeSeriesWindowDataset
from model.train_config import *

@dataclass
class DataConfig:
    # Used only to split windows; label_col is not used directly here
    train_ratio: float = 0.70
    val_ratio: float = 0.15

# -----------------------------
# Helpers
# -----------------------------
def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chrono_split_by_window_ends(M: int, tr_r: float, va_r: float) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    n_tr = int(M * tr_r)
    n_va = int(M * va_r)
    return (0, n_tr), (n_tr, n_tr + n_va), (n_tr + n_va, M)

def compute_minimum_size(y: np.ndarray,candidate_idx) -> dict[int, int]:
    pos_idx = candidate_idx[y[candidate_idx] == common.Signal.POSITIVE]
    neg_idx = candidate_idx[y[candidate_idx] == common.Signal.NEGATIVE]
    neu_idx = candidate_idx[y[candidate_idx] == common.Signal.NEUTRAL]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    n_neu = len(neu_idx)
    target_n = min(n_pos, n_neg, n_neu)

    if target_n <= 0:
        raise RuntimeError(
            f"Balanced test downsampling failed: pos={n_pos}, neg={n_neg}, neu={n_neu}"
        )
    return pos_idx,neg_idx,neu_idx,target_n


def sample_balanced_indices_downsample(
    logger: logging.Logger,
    y: np.ndarray,
    candidate_idx: np.ndarray,
    size:int,
    seed: int,
) -> np.ndarray:
    """
    Within candidate_idx, perform 3-class balanced downsampling according to current y.
    Each class is downsampled to min(pos, neg, neutral).
    """
    pos_idx, neg_idx, neu_idx, target_n = compute_minimum_size(y,candidate_idx)
    if size > target_n:
        raise RuntimeError(f"Requested balanced sample size {size} exceeds available {target_n} for pos/neg/neut.")

    rng = np.random.default_rng(seed)
    pos_s = rng.choice(pos_idx, size, replace=False)
    neg_s = rng.choice(neg_idx, size, replace=False)
    neu_s = rng.choice(neu_idx, size, replace=False)

    out = np.concatenate([pos_s, neg_s, neu_s])
    rng.shuffle(out)
    return out

def plot_cross_eval_heatmap(
    cross_df: pd.DataFrame,
    save_dir: str,
    metric: str,
    annotate_std: bool = True,
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

        ax.set_xlabel("Eval Label Threshold λ")
        ax.set_ylabel("Train Label Threshold λ")
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
) -> list[str]:

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    out_paths = []

    for model_name, df_one in self_summary_df.groupby("model", dropna=False):
        model_save_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)

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

        plt.title(
            f"Self Evaluation: {metric.upper()} Mean ± Std\n"
            f"Model={model_name}"
        )

        plt.xlabel("Label Threshold Multiplier λ")
        plt.ylabel(metric.upper())

        plt.grid(True, alpha=0.3)

        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(Line2D([], [], linestyle="none", label=info))
        plt.legend(handles=handles, loc="best", framealpha=0.9)

        plt.tight_layout()

        out_path = os.path.join(
            model_save_dir,
            f"self_eval_{metric}_mean_std.png",
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
):
    mean_col = f"{metric}_mean_mean"
    std_col = f"{metric}_mean_std"

    for (model, eval_mode), df_one in cross_train_batch_summary_df.groupby(
        ["model", "eval_mode"]
    ):
        model_save_dir = os.path.join(save_dir, model)
        os.makedirs(model_save_dir, exist_ok=True)

        df_plot = df_one.sort_values("train_threshold").copy()

        x = df_plot["train_threshold"].to_numpy()
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

        plt.xlabel("Train Threshold λ")
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
):
    mean_col = f"{metric}_mean_mean"
    std_col = f"{metric}_mean_std"

    for (model, eval_mode), df_one in cross_eval_batch_summary_df.groupby(
        ["model", "eval_mode"]
    ):
        model_save_dir = os.path.join(save_dir, model)
        os.makedirs(model_save_dir, exist_ok=True)

        df_plot = df_one.sort_values("eval_threshold").copy()

        x = df_plot["eval_threshold"].to_numpy()
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

        plt.xlabel("Eval Threshold λ")
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

def plot_train_vs_eval_summary(cross_train_batch_summary_df,cross_eval_batch_summary_df,save_dir,
                                metric: str,total_runs):
    mean_col = f"{metric}_mean_mean"

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

            train_size = int(train_df["train_size"].iloc[0])
            test_size = int(eval_df["test_size"].iloc[0])
            info = (
                f"{total_runs} independent runs\n"
                f"train samples = {train_size}\n"
                f"test samples = {test_size}"
            )
            fig, ax = plt.subplots(figsize=(12, 7))

            ax.plot(
                train_df["train_threshold"],
                train_df[mean_col],
                marker="o",
                linewidth=3,
                label="Train Threshold Robustness",
            )
            ax.plot(
                eval_df["eval_threshold"],
                eval_df[mean_col],
                marker="s",
                linewidth=3,
                linestyle="--",
                label="Eval Threshold Separability",
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

def prepare_parameter_regime_datasets(logger: logging.Logger, seed, labels_matrix, train_idx, test_idx, label_cols,strictest_train_target_n, strictest_test_target_n):
    experiment_datsets:dict[str,dict] = {}
    #get minimum size of POS/NEG for the strictest threshold
    y_strictest = labels_matrix[:, len(label_cols)-1].astype(np.int64)
    train_target_n = strictest_train_target_n
    test_target_n = strictest_test_target_n
    logger.warning("strictest label column is %s, with train target n=%d and test target n=%d", label_cols[-1], train_target_n, test_target_n)

    for col_idx, col_name in enumerate(label_cols):
        y_all = labels_matrix[:, col_idx].astype(np.int64)

        balanced_train_idx = sample_balanced_indices_downsample(
            logger,
            y=y_all,
            candidate_idx=train_idx,
            size=train_target_n,
            seed=seed + 10000 + col_idx*100,
        )

        balanced_test_idx = sample_balanced_indices_downsample(
            logger,
            y=y_all,
            candidate_idx=test_idx,
            size=test_target_n,
            seed=seed + 10000 + col_idx*100 + 1,
        )
        experiment_datsets[col_name] = {
            "balanced_train_idx": balanced_train_idx,
            "balanced_test_idx": balanced_test_idx,
        }
    return experiment_datsets

# -----------------------------
# Core experiment
# -----------------------------
def run_fixed_neutral_subsampling_experiment(
    logger: logging.Logger,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    pre_para: common.BaseDefine,
    prep_output_dir: str,
    save_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(train_cfg.seed)
    # warnings.filterwarnings("ignore", category=ConvergenceWarning)
    os.makedirs(save_dir, exist_ok=True)

    df = common.load_train_df_from_dir(prep_output_dir)
    if pre_para is None:
        pre_para = common.load_interval_ms_from_dir(prep_output_dir)
        logger.info(f"load pre_para from {prep_output_dir}")
    kline_interval_ms = common.get_interval_ms(pre_para.interval)

    label_cols = sorted(
        [c for c in df.columns if c.startswith("label_v")],
        key=lambda x: int(x.replace("label_v", "")),
    )
    if not label_cols:
        raise RuntimeError("No label_vxx columns found in df.")

    master_ds = TimeSeriesWindowDataset(
        df=df,
        kline_interval_ms=kline_interval_ms,
        feature_cols=train_cfg.feature_conf_list,
        label_col=label_cols[0],
        window=train_cfg.seq_len,
        stride=train_cfg.stride,
        use_cache=False,
        show_feature_distribution=False,
    )

    window_indices = master_ds.indices
    labels_matrix = df.loc[window_indices, label_cols].values.astype(np.int64)

    X_full = master_ds.X.detach().cpu().numpy()
    logger.info(f"X_full shape: {X_full.shape} | labels_matrix shape: {labels_matrix.shape}")
    M = X_full.shape[0]
    X_full_flat = X_full.reshape(M, -1)

    tr_rng, _, te_rng = chrono_split_by_window_ends(M, data_cfg.train_ratio, data_cfg.val_ratio)
    train_idx = np.arange(tr_rng[0], tr_rng[1])
    test_idx = np.arange(te_rng[0], te_rng[1])

    logger.info(f"Master windows M={M} | train={len(train_idx)} | test={len(test_idx)}")

    self_eval_rows = []
    cross_eval_rows = []
    total_runs = 20

    y_strictest = labels_matrix[:, len(label_cols)-1].astype(np.int64)
    _,_,_,strictest_train_target_n = compute_minimum_size(y_strictest,train_idx)
    _,_,_,strictest_test_target_n = compute_minimum_size(y_strictest,test_idx)
    for run_id in range(total_runs):
        logger.info(f"Starting run {run_id+1}/{total_runs}")
        t_seed = train_cfg.seed + run_id*10000
        experiment_datsets = prepare_parameter_regime_datasets(logger,t_seed, labels_matrix, train_idx, test_idx, label_cols,strictest_train_target_n,strictest_test_target_n)

        for experiment_model in ['DecisionTree','LogisticRegression']:
            #train
            for col_idx, col_name in enumerate(label_cols):
                logger.info(f"{experiment_model} train {col_name}")
                y_all = labels_matrix[:, col_idx].astype(np.int64)
                threshold = int(col_name.replace("label_v", "")) / 10.0

                # if eval_mode == 'balanced' or ['raw']:
                train_index = experiment_datsets[col_name]["balanced_train_idx"]
                n_eff = len(train_index) // 3
                test_index = experiment_datsets[col_name]["balanced_test_idx"]
                # elif eval_mode == 'raw':
                #     train_index = train_idx
                #     n_eff = len(train_index) // 3
                #     test_index = test_idx
                X_tr_flat = X_full_flat[train_index]
                y_tr = y_all[train_index]
                logger.info(f"Starting experiment: Model={experiment_model} | train_size={len(train_index)} (effective balanced size per class={n_eff}) | test_size={len(test_index)}")
                if experiment_model == 'LogisticRegression':
                    model = LogisticRegression(
                        solver=train_cfg.lr_solver,
                        max_iter=train_cfg.lr_max_iter,
                        C=train_cfg.lr_C,
                    )
                elif experiment_model == 'DecisionTree':
                    model = DecisionTreeClassifier(
                        min_samples_leaf=max(20, int(0.01 * len(y_tr))),
                        max_depth=5,
                        random_state=t_seed)
                model.fit(X_tr_flat, y_tr)

                # Within-Regime Evaluation: balanced test set
                X_te_bal = X_full_flat[test_index]
                y_te_bal = y_all[test_index]

                y_pred_bal = model.predict(X_te_bal)
                f1_bal = float(f1_score(y_te_bal, y_pred_bal, average="macro"))
                mcc_bal = float(matthews_corrcoef(y_te_bal, y_pred_bal))
                bal_acc = float(balanced_accuracy_score(y_te_bal, y_pred_bal))
                class_labels = [common.Signal.POSITIVE, common.Signal.NEGATIVE, common.Signal.NEUTRAL]
                p_class, r_class, f_class, _ = precision_recall_fscore_support(
                    y_te_bal, y_pred_bal, labels=class_labels, zero_division=0
                )

                n_pos = int(np.sum(y_all[test_idx] == common.Signal.POSITIVE))
                n_neg = int(np.sum(y_all[test_idx] == common.Signal.NEGATIVE))
                n_neu = int(np.sum(y_all[test_idx] == common.Signal.NEUTRAL))
                balanced_class_size = min(n_pos, n_neg, n_neu)

                logger.info(
                    f"{col_name} | raw test counts: pos={n_pos}, neg={n_neg}, neu={n_neu}, "
                    f"balanced_class_size={balanced_class_size}"
                )
                self_eval_rows.append(
                    {
                        "run_id": run_id,
                        "model": experiment_model,
                        "eval_mode": 'balanced',
                        "label_name": col_name,
                        "threshold": threshold,
                        "macro_f1": f1_bal,
                        "mcc": mcc_bal,
                        "balanced_accuracy": bal_acc,
                        "p_pos": p_class[0], "r_pos": r_class[0], "f_pos": f_class[0],
                        "p_neg": p_class[1], "r_neg": r_class[1], "f_neg": f_class[1],
                        "p_neu": p_class[2], "r_neu": r_class[2], "f_neu": f_class[2],
                        "n_eff": n_eff,
                        "train_size": int(3 * n_eff),
                        "test_size": int(len(y_te_bal)),
                        "test_pos_raw": n_pos,
                        "test_neg_raw": n_neg,
                        "test_neu_raw": n_neu,
                        "balanced_class_size": balanced_class_size,
                    }
                )

                # Cross-Regime Evaluation: evaluate on all test sets with different thresholds
                for eval_mode in ['balanced','raw']:
                    for eval_col_idx, eval_col_name in enumerate(label_cols):
                        y_all = labels_matrix[:, eval_col_idx].astype(np.int64)
                        eval_threshold = int(eval_col_name.replace("label_v", "")) / 10.0

                        if eval_mode == 'balanced':
                            eval_test_idx  = experiment_datsets[eval_col_name]["balanced_test_idx"]
                        elif eval_mode == 'raw':
                            eval_test_idx = test_idx
                        X_te_bal = X_full_flat[eval_test_idx ]
                        y_te_bal = y_all[eval_test_idx ] 
                        y_pred_cross  = model.predict(X_te_bal)

                        f1_cross = float(f1_score(y_te_bal, y_pred_cross , average="macro"))
                        mcc_bal = float(matthews_corrcoef(y_te_bal, y_pred_cross))

                        cross_eval_rows.append(
                            {
                                "run_id": run_id,
                                "model": experiment_model,
                                "train_size": int(3 * n_eff),
                                "eval_mode": eval_mode,
                                "train_label_name": col_name,
                                "train_threshold": threshold,
                                "eval_label_name": eval_col_name,
                                "eval_threshold": eval_threshold,
                                "macro_f1": f1_cross,
                                "mcc": mcc_bal,
                                "test_size": int(len(y_te_bal)),
                            }
                        )

    self_eval_df = pd.DataFrame(self_eval_rows)
    cross_eval_df = pd.DataFrame(cross_eval_rows)

    self_summary_df = (self_eval_df.groupby(["model", "threshold"]).agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            mcc_mean=("mcc", "mean"),
            mcc_std=("mcc", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            n_runs=("run_id", "nunique"),
            train_size=("train_size", "first"),
            test_size=("test_size", "first"),
        )
        .reset_index()
    )

    cross_train_summary_df = (cross_eval_df.groupby(["run_id", "model", "eval_mode", "train_threshold"]).agg(
            macro_f1_mean=("macro_f1", "mean"),
            mcc_mean=("mcc", "mean"),
            train_size=("train_size", "first"),
            test_size=("test_size", "first"),
        )
        .reset_index()
    )
    cross_train_batch_summary_df = (cross_train_summary_df.groupby(["model", "eval_mode", "train_threshold"]).agg(
            macro_f1_mean_mean=("macro_f1_mean", "mean"),
            macro_f1_mean_std=("macro_f1_mean", "std"),
            mcc_mean_mean=("mcc_mean", "mean"),
            mcc_mean_std=("mcc_mean", "std"),
            train_size=("train_size", "first"),
            test_size=("test_size", "first"),
        )
        .reset_index()
    )
    cross_eval_summary_df = (cross_eval_df.groupby(["run_id", "model", "eval_mode", "eval_threshold"]).agg(
            macro_f1_mean=("macro_f1", "mean"),
            mcc_mean=("mcc", "mean"),
            train_size=("train_size", "first"),
            test_size=("test_size", "first"),
        )
        .reset_index()
    )
    cross_eval_batch_summary_df = (cross_eval_summary_df.groupby(["model", "eval_mode", "eval_threshold"]).agg(
            macro_f1_mean_mean=("macro_f1_mean", "mean"),
            macro_f1_mean_std=("macro_f1_mean", "std"),
            mcc_mean_mean=("mcc_mean", "mean"),
            mcc_mean_std=("mcc_mean", "std"),
            train_size=("train_size", "first"),
            test_size=("test_size", "first"),
        )
        .reset_index()
    )

    cross_matrix_summary_df = (
        cross_eval_df
        .groupby(["model", "eval_mode", "train_threshold", "eval_threshold"])
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            mcc_mean=("mcc", "mean"),
            mcc_std=("mcc", "std"),
            n_runs=("run_id", "nunique"),
            train_size=("train_size", "first"),
            test_size=("test_size", "first"),
        )
        .reset_index()
    )

    self_eval_df.to_csv(os.path.join(save_dir, "self_eval_all_runs.csv"), index=False)
    cross_eval_df.to_csv(os.path.join(save_dir, "cross_eval_all_runs.csv"), index=False)
    self_summary_df.to_csv(os.path.join(save_dir, "self_eval_summary_mean_std.csv"), index=False)
    cross_train_summary_df.to_csv(os.path.join(save_dir, "cross_train_summary_mean_std.csv"), index=False)
    cross_train_batch_summary_df.to_csv(os.path.join(save_dir, "cross_train_batch_summary_mean_std.csv"), index=False)
    cross_eval_summary_df.to_csv(os.path.join(save_dir, "cross_eval_summary_std.csv"), index=False)
    cross_eval_batch_summary_df.to_csv(os.path.join(save_dir, "cross_eval_batch_summary_mean_std.csv"), index=False)

    plot_self_eval_mean_std(self_summary_df, save_dir, metric="macro_f1")
    plot_self_eval_mean_std(self_summary_df, save_dir, metric="balanced_accuracy")
    plot_self_eval_mean_std(self_summary_df, save_dir, metric="mcc")

    plot_cross_train_batch_summary(cross_train_batch_summary_df, save_dir, metric="macro_f1", total_runs=total_runs)
    plot_cross_train_batch_summary(cross_train_batch_summary_df, save_dir, metric="mcc", total_runs=total_runs)
    plot_cross_eval_batch_summary(cross_eval_batch_summary_df, save_dir, metric="macro_f1", total_runs=total_runs)
    plot_cross_eval_batch_summary(cross_eval_batch_summary_df, save_dir, metric="mcc", total_runs=total_runs)
    plot_train_vs_eval_summary(cross_train_batch_summary_df, cross_eval_batch_summary_df, save_dir, metric="macro_f1", total_runs=total_runs)
    plot_train_vs_eval_summary(cross_train_batch_summary_df, cross_eval_batch_summary_df, save_dir, metric="mcc", total_runs=total_runs)
    plot_cross_eval_heatmap(cross_matrix_summary_df, save_dir, metric="macro_f1")
    plot_cross_eval_heatmap(cross_matrix_summary_df, save_dir, metric="mcc")

# -----------------------------
# Entrypoint
# -----------------------------
def main(
    logger: logging.Logger,
    train_cfg: TrainConfig = TrainConfig(),
    pre_para: common.BaseDefine = common.BaseDefine(),
    prep_output_dir: str = common.DATA_OUT_DIR,
    save_dir: str = common.TRAIN_OUT_DIR,
)   -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(save_dir, exist_ok=True)
    data_cfg = DataConfig()
    run_fixed_neutral_subsampling_experiment(
        logger=logger,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        pre_para=pre_para,
        prep_output_dir=prep_output_dir,
        save_dir=save_dir,
    )

if __name__ == "__main__":
    logger, _ = common.setup_session_logger(sub_folder="train", file_level=logging.DEBUG)

    begin_time = time.time()
    prep_output_dir = common.DATA_OUT_DIR

    cfg = TrainConfig()

    main(logger, train_cfg=cfg, pre_para=None, prep_output_dir=prep_output_dir)

    end_time = time.time()
    logger.info(f"Total time: {(end_time - begin_time):.2f} seconds")