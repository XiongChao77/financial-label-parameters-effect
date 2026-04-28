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
from sklearn.metrics import f1_score
from sklearn.exceptions import ConvergenceWarning
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
    else:
        logger.info(f"sample numble for downsample {target_n}")
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

def plot_self_balanced_eval_curve(results_df: pd.DataFrame, save_dir: str) -> str:
    import matplotlib.pyplot as plt

    df_plot = results_df.sort_values("threshold").reset_index(drop=True)
    x = df_plot["threshold"].to_numpy()
    y = df_plot["macro_f1"].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o", linewidth=2)

    best_i = int(np.argmax(y))
    plt.axvline(x=x[best_i], linestyle="--", alpha=0.6)
    plt.annotate(
        f"Best\nλ={x[best_i]:.1f}\nF1={y[best_i]:.4f}",
        xy=(x[best_i], y[best_i]),
        xytext=(x[best_i] + 0.2, y[best_i]),
        arrowprops=dict(arrowstyle="->"),
    )

    plt.title("Self Evaluation on Balanced Test Set")
    plt.xlabel("Label Threshold Multiplier (λ)")
    plt.ylabel("Macro-F1")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    out_path = os.path.join(save_dir, "self_balanced_eval_curve.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def plot_cross_eval_heatmap(cross_df: pd.DataFrame, save_dir: str) -> str:
    import matplotlib.pyplot as plt

    pivot = cross_df.pivot(
        index="train_threshold",
        columns="eval_threshold",
        values="macro_f1",
    ).sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.1f}" for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{x:.1f}" for x in pivot.index])

    ax.set_xlabel("Eval Label Threshold")
    ax.set_ylabel("Train Label Threshold")
    ax.set_title("Cross Evaluation Macro-F1")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    out_path = os.path.join(save_dir, "cross_eval_macro_f1_heatmap.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def prepare_parameter_regime_datasets(train_cfg, labels_matrix, train_idx, test_idx, label_cols):
    experiment_dasets:dict[str,dict] = {}
    #get minimum size of POS/NEG for the strictest threshold
    y_strictest = labels_matrix[:, len(label_cols)-1].astype(np.int64)
    train_pos_idx, train_neg_idx, train_neu_idx, train_target_n = compute_minimum_size(y_strictest,train_idx)
    test_pos_idx, test_neg_idx, test_neu_idx, test_target_n = compute_minimum_size(y_strictest,test_idx)

    for col_idx, col_name in enumerate(label_cols):
        y_all = labels_matrix[:, col_idx].astype(np.int64)

        balanced_train_idx = sample_balanced_indices_downsample(
            logger,
            y=y_all,
            candidate_idx=train_idx,
            size=train_target_n,
            seed=train_cfg.seed + 10000 + col_idx,
        )

        print(f"balanced_train_idx for {col_name} is {len(balanced_train_idx)}")

        balanced_test_idx = sample_balanced_indices_downsample(
            logger,
            y=y_all,
            candidate_idx=test_idx,
            size=test_target_n,
            seed=train_cfg.seed + 10000 + col_idx,
        )
        experiment_dasets[col_name] = {
            "balanced_train_idx": balanced_train_idx,
            "balanced_test_idx": balanced_test_idx,
        }
    return experiment_dasets

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
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
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
    print(f"X_full shape: {X_full.shape} | labels_matrix shape: {labels_matrix.shape}")
    M = X_full.shape[0]
    X_full_flat = X_full.reshape(M, -1)

    tr_rng, _, te_rng = chrono_split_by_window_ends(M, data_cfg.train_ratio, data_cfg.val_ratio)
    train_idx = np.arange(tr_rng[0], tr_rng[1])
    test_idx = np.arange(te_rng[0], te_rng[1])

    logger.info(f"Master windows M={M} | train={len(train_idx)} | test={len(test_idx)}")

    experiment_dasets = prepare_parameter_regime_datasets(train_cfg, labels_matrix, train_idx, test_idx, label_cols)

    X_te_flat = X_full_flat[test_idx]

    test_label_map = {}
    for col_idx, col_name in enumerate(label_cols):
        test_label_map[col_name] = labels_matrix[test_idx, col_idx].astype(np.int64)

    self_eval_rows = []
    cross_eval_rows = []

    #train
    for col_idx, col_name in enumerate(label_cols):
        logger.info(f"train {col_name}")
        y_all = labels_matrix[:, col_idx].astype(np.int64)
        threshold = int(col_name.replace("label_v", "")) / 10.0

        train_balanced_idx = experiment_dasets[col_name]["balanced_train_idx"]
        n_eff = len(train_balanced_idx) // 3
        balanced_test_idx = experiment_dasets[col_name]["balanced_test_idx"]
        X_tr_flat = X_full_flat[train_balanced_idx]
        y_tr = y_all[train_balanced_idx]
        # model = LogisticRegression(
        #     solver=train_cfg.lr_solver,
        #     max_iter=train_cfg.lr_max_iter,
        #     C=train_cfg.lr_C,
        # )
        model = DecisionTreeClassifier(
            min_samples_leaf=max(20, int(0.01 * len(y_tr))),
            max_depth=5,
            random_state=train_cfg.seed)
        model.fit(X_tr_flat, y_tr)

        # Within-Regime Evaluation: balanced test set
        X_te_bal = X_full_flat[balanced_test_idx]    #balanced_test_idx
        y_te_bal = y_all[balanced_test_idx]

        y_pred_bal = model.predict(X_te_bal)
        f1_bal = float(f1_score(y_te_bal, y_pred_bal, average="macro"))

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
                "label_name": col_name,
                "threshold": threshold,
                "macro_f1": f1_bal,
                "n_eff": n_eff,
                "train_size": int(3 * n_eff),
                "test_size": int(len(y_te_bal)),
                "test_pos_raw": n_pos,
                "test_neg_raw": n_neg,
                "test_neu_raw": n_neu,
                "balanced_class_size": balanced_class_size,
            }
        )

        # Cross-Regime Evaluation: evaluate on all test sets with different thresholds (balanced)
        logger.info(f"Cross evaluation for {col_name} on all test thresholds")
        for eval_col_idx, eval_col_name in enumerate(label_cols):
            y_all = labels_matrix[:, eval_col_idx].astype(np.int64)
            eval_threshold = int(eval_col_name.replace("label_v", "")) / 10.0
            y_te_cross = test_label_map[eval_col_name]

            eval_balanced_test_idx  = experiment_dasets[eval_col_name]["balanced_test_idx"]
            X_te_bal = X_full_flat[eval_balanced_test_idx ]
            y_te_bal = y_all[eval_balanced_test_idx ] 
            y_pred_cross  = model.predict(X_te_bal)

            f1_cross = float(f1_score(y_te_bal, y_pred_cross , average="macro"))

            cross_eval_rows.append(
                {
                    "train_label_name": eval_col_name,
                    "train_threshold": threshold,
                    "eval_label_name": eval_col_name,
                    "eval_threshold": eval_threshold,
                    "macro_f1": f1_cross,
                    "test_size": int(len(y_te_bal)),
                }
            )

        logger.info(f"Done {col_name} | self-balanced F1={f1_bal:.4f}")

    self_eval_df = pd.DataFrame(self_eval_rows).sort_values("threshold").reset_index(drop=True)
    cross_eval_df = pd.DataFrame(cross_eval_rows).sort_values(
        ["train_threshold", "eval_threshold"]
    ).reset_index(drop=True)

    self_csv = os.path.join(save_dir, "self_balanced_eval.csv")
    cross_csv = os.path.join(save_dir, "cross_eval_balanced.csv")
    self_eval_df.to_csv(self_csv, index=False)
    cross_eval_df.to_csv(cross_csv, index=False)

    fig1 = plot_self_balanced_eval_curve(self_eval_df, save_dir)
    fig2 = plot_cross_eval_heatmap(cross_eval_df, save_dir)

    logger.info(f"Saved self eval csv: {self_csv}")
    logger.info(f"Saved cross eval csv: {cross_csv}")
    logger.info(f"Saved self eval figure: {fig1}")
    logger.info(f"Saved cross eval heatmap: {fig2}")

    return self_eval_df, cross_eval_df

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
    return run_fixed_neutral_subsampling_experiment(
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