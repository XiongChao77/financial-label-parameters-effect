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

import os,sys,re
import time
import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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
from model.models import lstm

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

def compute_financial_returns(
    df: pd.DataFrame,
    pre_para: common.BaseDefine,
    para_horizon:int,
    window_indices: np.ndarray,
    sample_idx: np.ndarray,
    y_pred: np.ndarray,
    label_col: str,
    initial_cash: float = 10000.0,
    fee_rates: list[float] | tuple[float, ...] = (0.0, 0.0005),
    price_col: str = "close",
) -> list[dict]:
    if pre_para.label_type == 'FTHL':
        return compute_financial_returns_with_exit_index_vectorized(df,window_indices,sample_idx,y_pred,label_col,
                                                            initial_cash,fee_rates,price_col)
    elif pre_para.label_type == 'TBM':
        return compute_financial_returns_tbm(df,para_horizon, window_indices,sample_idx,y_pred,label_col,
                                                            initial_cash,fee_rates,price_col)

def compute_financial_returns_tbm(
    df: pd.DataFrame,
    para_horizon: int,
    window_indices: np.ndarray,
    sample_idx: np.ndarray,
    y_pred: np.ndarray,
    label_col: str,
    initial_cash: float = 10000.0,
    fee_rates: list[float] | tuple[float, ...] = (0.0, 0.0005),
    price_col: str = "close",
) -> list[dict]:
    """
    TBM financial return diagnostics.

    Trading logic:
    - y_pred == POSITIVE: long
    - y_pred == NEGATIVE: short
    - y_pred == NEUTRAL : no trade

    Exit logic:
    - Iterate K-lines after entry.
    - Only use K-lines whose open_time_sn <= entry_open_time_sn + para_horizon.
    - Stop-loss has priority over take-profit.
    - If neither TP nor SL is touched inside the horizon, exit at the close price
      of the last valid K-line inside the horizon.

    Metrics:
    1. Signal-level:
       - every predicted long/short signal is treated as an independent trade
       - overlapping trades are allowed
       - same notional = initial_cash for each signal

    2. Strategy-level:
       - non-overlapping trades only
       - full-capital compounding
       - overlapping entries are skipped

    Fee convention:
    - fee_rate is one-way transaction fee
    - round_trip_fee = 2 * fee_rate
    """

    if para_horizon <= 0:
        raise ValueError(f"para_horizon must be positive, got {para_horizon}")

    required_cols = [
        price_col,
        "high",
        "low",
        "open_time_sn",
        f"{label_col}_threshold_long",
        f"{label_col}_threshold_short",
        f"{label_col}_stop_threshold_long",
        f"{label_col}_stop_threshold_short",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for TBM return: {missing}")

    prices = df[price_col].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    open_time_sn = df["open_time_sn"].to_numpy(dtype=np.int64)

    threshold_long = df[f"{label_col}_threshold_long"].to_numpy(dtype=float)
    threshold_short = df[f"{label_col}_threshold_short"].to_numpy(dtype=float)
    stop_threshold_long = df[f"{label_col}_stop_threshold_long"].to_numpy(dtype=float)
    stop_threshold_short = df[f"{label_col}_stop_threshold_short"].to_numpy(dtype=float)

    entry_idx = window_indices[sample_idx].astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    if len(entry_idx) != len(y_pred):
        raise ValueError(
            f"entry_idx and y_pred length mismatch: {len(entry_idx)} vs {len(y_pred)}"
        )

    fee_rates = list(fee_rates)
    if len(fee_rates) == 0:
        raise ValueError("fee_rates must contain at least one value.")
    if any(f < 0 for f in fee_rates):
        raise ValueError(f"fee_rates must be non-negative: {fee_rates}")

    n = len(df)

    valid_entry = (
        (entry_idx >= 0)
        & (entry_idx < n)
        & np.isfinite(prices[entry_idx])
        & (prices[entry_idx] > 0)
    )

    entry_idx = entry_idx[valid_entry]
    y_valid = y_pred[valid_entry]

    trade_mask = (
        (y_valid == common.Signal.POSITIVE)
        | (y_valid == common.Signal.NEGATIVE)
    )

    raw_trade_entry_idx = entry_idx[trade_mask]
    raw_trade_signal = y_valid[trade_mask]

    trade_entry_idx = []
    trade_exit_idx = []
    trade_signal = []
    gross_signal_ret = []
    trade_exit_reason = []

    for ent_i, sig in zip(raw_trade_entry_idx, raw_trade_signal):
        ent_i = int(ent_i)
        sig = int(sig)

        entry_price = float(prices[ent_i])
        entry_sn = int(open_time_sn[ent_i])
        max_exit_sn = entry_sn + int(para_horizon)

        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        # No future K-line.
        if ent_i + 1 >= n:
            continue

        ex_i = None
        ex_price = None
        reason = None
        last_valid_i = None

        # =====================================================
        # Long trade
        # =====================================================
        if sig == common.Signal.POSITIVE:
            tp_rate = float(threshold_long[ent_i])
            sl_rate = float(stop_threshold_long[ent_i])

            if not np.isfinite(tp_rate):
                continue

            tp_price = entry_price * (1.0 + tp_rate)

            if np.isfinite(sl_rate):
                sl_price = entry_price * (1.0 - sl_rate)
            else:
                sl_price = -np.inf

            for j in range(ent_i + 1, n):
                curr_sn = int(open_time_sn[j])

                # Important:
                # Use open_time_sn to decide whether this K-line is still
                # inside the TBM horizon.
                if curr_sn > max_exit_sn:
                    break

                last_valid_i = j

                h = float(highs[j])
                l = float(lows[j])

                if not np.isfinite(h) or not np.isfinite(l):
                    continue

                hit_sl = l <= sl_price
                hit_tp = h >= tp_price

                # Stop-loss priority.
                # If TP and SL are touched in the same K-line, SL wins.
                if hit_sl:
                    ex_i = j
                    ex_price = sl_price
                    reason = "long_sl"
                    break

                if hit_tp:
                    ex_i = j
                    ex_price = tp_price
                    reason = "long_tp"
                    break

            if ex_i is None:
                if last_valid_i is None:
                    continue

                ex_i = int(last_valid_i)
                ex_price = float(prices[ex_i])
                reason = "long_timeout"

            if not np.isfinite(ex_price) or ex_price <= 0:
                continue

            trade_return = ex_price / entry_price - 1.0

        # =====================================================
        # Short trade
        # =====================================================
        elif sig == common.Signal.NEGATIVE:
            tp_rate = float(threshold_short[ent_i])
            sl_rate = float(stop_threshold_short[ent_i])

            if not np.isfinite(tp_rate):
                continue

            tp_price = entry_price * (1.0 - tp_rate)

            if np.isfinite(sl_rate):
                sl_price = entry_price * (1.0 + sl_rate)
            else:
                sl_price = np.inf

            for j in range(ent_i + 1, n):
                curr_sn = int(open_time_sn[j])

                if curr_sn > max_exit_sn:
                    break

                last_valid_i = j

                h = float(highs[j])
                l = float(lows[j])

                if not np.isfinite(h) or not np.isfinite(l):
                    continue

                hit_sl = h >= sl_price
                hit_tp = l <= tp_price

                # Stop-loss priority.
                if hit_sl:
                    ex_i = j
                    ex_price = sl_price
                    reason = "short_sl"
                    break

                if hit_tp:
                    ex_i = j
                    ex_price = tp_price
                    reason = "short_tp"
                    break

            if ex_i is None:
                if last_valid_i is None:
                    continue

                ex_i = int(last_valid_i)
                ex_price = float(prices[ex_i])
                reason = "short_timeout"

            if not np.isfinite(ex_price) or ex_price <= 0:
                continue

            # Short return.
            trade_return = entry_price / ex_price - 1.0

        else:
            continue

        trade_entry_idx.append(ent_i)
        trade_exit_idx.append(int(ex_i))
        trade_signal.append(sig)
        gross_signal_ret.append(float(trade_return))
        trade_exit_reason.append(reason)

    trade_entry_idx = np.asarray(trade_entry_idx, dtype=np.int64)
    trade_exit_idx = np.asarray(trade_exit_idx, dtype=np.int64)
    trade_signal = np.asarray(trade_signal, dtype=np.int64)
    gross_signal_ret = np.asarray(gross_signal_ret, dtype=float)

    results = []

    for fee_rate in fee_rates:
        fee_rate = float(fee_rate)
        round_trip_fee = 2.0 * fee_rate

        net_signal_ret = gross_signal_ret - round_trip_fee

        # =====================================================
        # 1. Signal-level independent return
        # =====================================================
        signal_trade_count = int(len(net_signal_ret))

        if signal_trade_count > 0:
            signal_avg_return = float(np.mean(net_signal_ret))
            signal_total_pnl = float(np.sum(initial_cash * net_signal_ret))
            signal_avg_pnl = float(np.mean(initial_cash * net_signal_ret))
            signal_win_rate = float(np.mean(net_signal_ret > 0))
        else:
            signal_avg_return = np.nan
            signal_total_pnl = 0.0
            signal_avg_pnl = np.nan
            signal_win_rate = np.nan

        long_ret = net_signal_ret[trade_signal == common.Signal.POSITIVE]
        short_ret = net_signal_ret[trade_signal == common.Signal.NEGATIVE]

        # =====================================================
        # 2. Strategy-level non-overlapping path return
        # =====================================================
        cash = float(initial_cash)
        strategy_trade_returns = []

        active_until_sn = -1
        skipped_overlap = 0

        order = np.argsort(trade_entry_idx)

        for k in order:
            ent_i = int(trade_entry_idx[k])
            ex_i = int(trade_exit_idx[k])

            ent_sn = int(open_time_sn[ent_i])
            ex_sn = int(open_time_sn[ex_i])

            # Use open_time_sn to judge overlap.
            # ent_sn < active_until_sn means this signal appears before the
            # previous accepted trade exits.
            if ent_sn < active_until_sn:
                skipped_overlap += 1
                continue

            r = float(net_signal_ret[k])

            if 1.0 + r <= 0:
                cash = 0.0
                strategy_trade_returns.append(r)
                active_until_sn = ex_sn
                break

            cash *= 1.0 + r
            strategy_trade_returns.append(r)
            active_until_sn = ex_sn

        strategy_trade_count = int(len(strategy_trade_returns))

        if strategy_trade_count > 0:
            strategy_arr = np.asarray(strategy_trade_returns, dtype=float)
            strategy_total_return = float(cash / initial_cash - 1.0)
            strategy_total_pnl = float(cash - initial_cash)
            strategy_avg_trade_return = float(np.mean(strategy_arr))
            strategy_win_rate = float(np.mean(strategy_arr > 0))
        else:
            strategy_total_return = 0.0
            strategy_total_pnl = 0.0
            strategy_avg_trade_return = np.nan
            strategy_win_rate = np.nan

        results.append(
            {
                "initial_cash": float(initial_cash),
                "fee_rate": fee_rate,
                "round_trip_fee": float(round_trip_fee),
                "label_col": label_col,
                "para_horizon": int(para_horizon),

                "valid_sample_count": int(len(y_valid)),
                "raw_signal_trade_count": int(signal_trade_count),

                # strategy-level
                "final_cash": float(cash),
                "strategy_total_return": strategy_total_return,
                "strategy_total_pnl": strategy_total_pnl,
                "strategy_trade_count": strategy_trade_count,
                "strategy_avg_trade_return": strategy_avg_trade_return,
                "strategy_win_rate": strategy_win_rate,
                "strategy_skipped_overlap": int(skipped_overlap),

                # signal-level
                "signal_trade_count": signal_trade_count,
                "signal_avg_return": signal_avg_return,
                "signal_total_pnl": signal_total_pnl,
                "signal_avg_pnl": signal_avg_pnl,
                "signal_win_rate": signal_win_rate,

                # long / short breakdown
                "signal_long_count": int(len(long_ret)),
                "signal_short_count": int(len(short_ret)),
                "signal_long_avg_return": (
                    float(np.mean(long_ret)) if len(long_ret) > 0 else np.nan
                ),
                "signal_short_avg_return": (
                    float(np.mean(short_ret)) if len(short_ret) > 0 else np.nan
                ),

                # TBM diagnostics
                "tbm_tp_count": int(
                    sum(reason.endswith("_tp") for reason in trade_exit_reason)
                ),
                "tbm_sl_count": int(
                    sum(reason.endswith("_sl") for reason in trade_exit_reason)
                ),
                "tbm_timeout_count": int(
                    sum(reason.endswith("_timeout") for reason in trade_exit_reason)
                ),
            }
        )

    return results

def compute_financial_returns_with_exit_index_vectorized(
    df: pd.DataFrame,
    window_indices: np.ndarray,
    sample_idx: np.ndarray,
    y_pred: np.ndarray,
    label_col: str,
    initial_cash: float = 10000.0,
    fee_rates: list[float] | tuple[float, ...] = (0.0, 0.0005),
    price_col: str = "close",
) -> list[dict]:
    """
    Compute two financial return diagnostics using precomputed exit metadata.

    1. Strategy-level return:
       - initial capital = initial_cash
       - POSITIVE -> long
       - NEGATIVE -> short
       - NEUTRAL  -> no trade
       - hold until precomputed exit_index
       - non-overlapping trades only
       - full capital compounding

    2. Signal-level return:
       - every POSITIVE / NEGATIVE signal is treated as an independent trade
       - same notional = initial_cash for each signal
       - overlapping trades are allowed
       - used as signal quality diagnostic

    Fee convention:
       fee_rate is one-way transaction fee.
       round_trip_fee = 2 * fee_rate.
    """

    exit_price_col = f"{label_col}_exit_price"
    exit_index_col = f"{label_col}_exit_index"

    if price_col not in df.columns:
        raise ValueError(f"Missing price column: {price_col}")
    if exit_price_col not in df.columns:
        raise ValueError(f"Missing exit price column: {exit_price_col}")
    if exit_index_col not in df.columns:
        raise ValueError(f"Missing exit index column: {exit_index_col}")

    prices = df[price_col].to_numpy(dtype=float)
    exit_prices_all = df[exit_price_col].to_numpy(dtype=float)
    exit_indices_all = df[exit_index_col].to_numpy(dtype=np.int64)

    entry_idx = window_indices[sample_idx].astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    if len(entry_idx) != len(y_pred):
        raise ValueError(
            f"entry_idx and y_pred length mismatch: {len(entry_idx)} vs {len(y_pred)}"
        )

    fee_rates = list(fee_rates)
    if len(fee_rates) == 0:
        raise ValueError("fee_rates must contain at least one value.")
    if any(f < 0 for f in fee_rates):
        raise ValueError(f"fee_rates must be non-negative: {fee_rates}")

    # entry_idx should normally be valid because it comes from dataset window indices.
    # Keep this check for safety.
    valid_entry = (
        (entry_idx >= 0)
        & (entry_idx < len(prices))
        & (entry_idx < len(exit_prices_all))
        & (entry_idx < len(exit_indices_all))
    )

    entry_idx = entry_idx[valid_entry]
    y_pred = y_pred[valid_entry]

    entry_price = prices[entry_idx]
    exit_price = exit_prices_all[entry_idx]
    exit_idx = exit_indices_all[entry_idx]

    # exit_price = NaN means invalid future target / missing K-line / out-of-bound.
    valid = (
        np.isfinite(entry_price)
        & np.isfinite(exit_price)
        & (entry_price > 0)
        & (exit_price > 0)
        & (exit_idx >= 0)
    )

    entry_idx = entry_idx[valid]
    exit_idx = exit_idx[valid]
    entry_price = entry_price[valid]
    exit_price = exit_price[valid]
    y_valid = y_pred[valid]

    asset_ret = exit_price / entry_price - 1.0

    long_mask = y_valid == common.Signal.POSITIVE
    short_mask = y_valid == common.Signal.NEGATIVE
    trade_mask = long_mask | short_mask

    trade_entry_idx = entry_idx[trade_mask]
    trade_exit_idx = exit_idx[trade_mask]
    trade_signal = y_valid[trade_mask]
    trade_asset_ret = asset_ret[trade_mask]

    gross_signal_ret = np.where(
        trade_signal == common.Signal.POSITIVE,
        trade_asset_ret,
        -trade_asset_ret,
    )

    results = []

    for fee_rate in fee_rates:
        fee_rate = float(fee_rate)
        round_trip_fee = 2.0 * fee_rate

        net_signal_ret = gross_signal_ret - round_trip_fee

        # =====================================================
        # 1. Signal-level independent return
        # =====================================================
        signal_trade_count = int(len(net_signal_ret))

        if signal_trade_count > 0:
            signal_avg_return = float(np.mean(net_signal_ret))
            signal_total_pnl = float(np.sum(initial_cash * net_signal_ret))
            signal_avg_pnl = float(np.mean(initial_cash * net_signal_ret))
            signal_win_rate = float(np.mean(net_signal_ret > 0))
        else:
            signal_avg_return = np.nan
            signal_total_pnl = 0.0
            signal_avg_pnl = np.nan
            signal_win_rate = np.nan

        long_ret = net_signal_ret[trade_signal == common.Signal.POSITIVE]
        short_ret = net_signal_ret[trade_signal == common.Signal.NEGATIVE]

        # =====================================================
        # 2. Strategy-level non-overlapping path return
        # =====================================================
        cash = float(initial_cash)
        strategy_trade_returns = []

        active_until = -1
        skipped_overlap = 0

        order = np.argsort(trade_entry_idx)

        for k in order:
            ent_i = int(trade_entry_idx[k])
            ex_i = int(trade_exit_idx[k])

            if ent_i < active_until:
                skipped_overlap += 1
                continue

            r = float(net_signal_ret[k])

            if 1.0 + r <= 0:
                cash = 0.0
                strategy_trade_returns.append(r)
                active_until = ex_i
                break

            cash *= 1.0 + r
            strategy_trade_returns.append(r)
            active_until = ex_i

        strategy_trade_count = int(len(strategy_trade_returns))

        if strategy_trade_count > 0:
            strategy_arr = np.asarray(strategy_trade_returns, dtype=float)
            strategy_total_return = float(cash / initial_cash - 1.0)
            strategy_total_pnl = float(cash - initial_cash)
            strategy_avg_trade_return = float(np.mean(strategy_arr))
            strategy_win_rate = float(np.mean(strategy_arr > 0))
        else:
            strategy_total_return = 0.0
            strategy_total_pnl = 0.0
            strategy_avg_trade_return = np.nan
            strategy_win_rate = np.nan

        results.append(
            {
                "initial_cash": float(initial_cash),
                "fee_rate": fee_rate,
                "round_trip_fee": float(round_trip_fee),
                "label_col": label_col,

                "valid_sample_count": int(len(y_valid)),
                "raw_signal_trade_count": int(signal_trade_count),

                # strategy-level
                "final_cash": float(cash),
                "strategy_total_return": strategy_total_return,
                "strategy_total_pnl": strategy_total_pnl,
                "strategy_trade_count": strategy_trade_count,
                "strategy_avg_trade_return": strategy_avg_trade_return,
                "strategy_win_rate": strategy_win_rate,
                "strategy_skipped_overlap": int(skipped_overlap),

                # signal-level
                "signal_trade_count": signal_trade_count,
                "signal_avg_return": signal_avg_return,
                "signal_total_pnl": signal_total_pnl,
                "signal_avg_pnl": signal_avg_pnl,
                "signal_win_rate": signal_win_rate,

                # long / short breakdown
                "signal_long_count": int(len(long_ret)),
                "signal_short_count": int(len(short_ret)),
                "signal_long_avg_return": (
                    float(np.mean(long_ret)) if len(long_ret) > 0 else np.nan
                ),
                "signal_short_avg_return": (
                    float(np.mean(short_ret)) if len(short_ret) > 0 else np.nan
                ),
            }
        )

    return results

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

        out_path = os.path.join(
            model_save_dir,
            f"self_eval_{metric}_mean_std.png",
        )

        plt.savefig(out_path, dpi=250)
        plt.close()

        out_paths.append(out_path)

    return out_paths

def plot_financial_return_mean_std(
    financial_summary_df: pd.DataFrame,
    save_dir: str,
    metric: str,
    para_type:str = "volatility",
) -> list[str]:
    """
    Plot threshold-financial metric mean/std curves.

    Expected columns:
        model
        fee_rate
        threshold
        {metric}_mean
        {metric}_std
        n_runs
        test_size

    Example metrics:
        strategy_total_return
        signal_avg_return
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

    for (model_name, fee_rate), df_one in financial_summary_df.groupby(
        ["model", "fee_rate"],
        dropna=False,
    ):
        model_save_dir = os.path.join(save_dir, str(model_name), "financial_return")
        os.makedirs(model_save_dir, exist_ok=True)

        df_plot = df_one.sort_values("threshold").copy()

        x = df_plot["threshold"].to_numpy(dtype=float)
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

        # Mark best point.
        # 对收益类指标，越大越好；如果全是 nan，则跳过标注。
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

        plt.title(
            f"Financial Diagnostic: {metric}\n"
            f"Model={model_name}, Fee={fee_rate:.6f}"
        )

        if para_type == 'volatility':
            plt.xlabel("Label Threshold Multiplier λ")
        elif para_type == 'horizon':
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

def plot_cross_train_batch_summary(
    cross_train_batch_summary_df,
    save_dir,
    metric: str,
    total_runs,
    para_type:str = "volatility",
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

def plot_train_vs_eval_summary(cross_train_batch_summary_df,cross_eval_batch_summary_df,save_dir,
                                metric: str,total_runs,para_type:str = "volatility",):
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

        if para_type == 'volatility':
            label = "Threshold"
        elif para_type == 'horizon':    
            label = "Bar horizon"

        ax.plot(
            train_df["train_threshold"],
            train_df[mean_col],
            marker="o",
            linewidth=3,
            label=f"Train {label} Robustness",
        )
        ax.plot(
            eval_df["eval_threshold"],
            eval_df[mean_col],
            marker="s",
            linewidth=3,
            linestyle="--",
            label=f"Eval {label} Separability",
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

def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    num_classes: int = 3,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    input_size = X_train.shape[2]

    model = lstm.LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_ds)

    return model


def predict_lstm_model(
    model: nn.Module,
    X_test: np.ndarray,
    batch_size: int = 256,
):
    device = next(model.parameters()).device

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    test_loader = DataLoader(X_test_t, batch_size=batch_size, shuffle=False)

    preds = []

    model.eval()

    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.cpu().numpy())

    return np.concatenate(preds)

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
):
    set_seed(train_cfg.seed)
    # warnings.filterwarnings("ignore", category=ConvergenceWarning)

    df = common.load_train_df_from_dir(prep_output_dir)
    if pre_para is None:
        pre_para = common.load_interval_ms_from_dir(prep_output_dir)
        logger.info(f"load pre_para from {prep_output_dir}")
    save_dir = os.path.join(save_dir, f"{pre_para.symbol}_{pre_para.interval}",pre_para.label_type,pre_para.para_type)
    os.makedirs(save_dir, exist_ok=True)
    kline_interval_ms = common.get_interval_ms(pre_para.interval)

    if pre_para.para_type == 'volatility':
        label_suffix = "label_v"
    elif pre_para.para_type == 'horizon':
        label_suffix = "label_h"
    label_pattern = re.compile(rf"^{label_suffix}(\d+)$")

    label_cols = sorted(
        [c for c in df.columns if label_pattern.fullmatch(c)],
        key=lambda x: int(label_pattern.fullmatch(x).group(1)),
    )
    if not label_cols:
        raise RuntimeError(f"No {label_suffix}xx columns found in df.")

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
    financial_metrics_list = []
    total_runs = 20

    strictest_train_target_n = np.inf
    strictest_test_target_n = np.inf
    for col_idx, col_name in enumerate(label_cols):
        y_all = labels_matrix[:, col_idx].astype(np.int64)
        _,_,_,train_target_n = compute_minimum_size(y_all,train_idx)
        _,_,_,test_target_n = compute_minimum_size(y_all,test_idx)
        if train_target_n < strictest_train_target_n:
            strictest_train_target_n = train_target_n
            logger.info(f"New strictest train target n={strictest_train_target_n} found at column {col_name}")
        if test_target_n < strictest_test_target_n:
            strictest_test_target_n = test_target_n
            logger.info(f"New strictest test target n={strictest_test_target_n} found at column {col_name}")

    for run_id in range(total_runs):
        logger.info(f"Starting run {run_id+1}/{total_runs}")
        t_seed = train_cfg.seed + run_id*10000
        experiment_datsets = prepare_parameter_regime_datasets(logger,t_seed, labels_matrix, train_idx, test_idx, label_cols,strictest_train_target_n,strictest_test_target_n)

        for experiment_model in ['LSTM']: #['DecisionTree', 'LogisticRegression', 'LSTM']:
            #train
            for col_idx, col_name in enumerate(label_cols):
                logger.info(f"{experiment_model} train {col_name}")
                y_all = labels_matrix[:, col_idx].astype(np.int64)
                if pre_para.para_type == 'volatility':
                    threshold = int(col_name.replace(label_suffix, "")) / 10.0
                    para_horizon = pre_para.predict_num
                elif pre_para.para_type == 'horizon':
                    threshold = int(col_name.replace(label_suffix, ""))
                    para_horizon = threshold

                # if eval_mode == 'balanced' or ['raw']:
                train_index = experiment_datsets[col_name]["balanced_train_idx"]
                n_eff = len(train_index) // 3
                test_index = experiment_datsets[col_name]["balanced_test_idx"]
                # elif eval_mode == 'raw':
                #     train_index = train_idx
                #     n_eff = len(train_index) // 3
                #     test_index = test_idx
                X_tr_flat = X_full_flat[train_index]
                X_tr_seq = X_full[train_index]
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
                elif experiment_model == 'LSTM':
                    model = train_lstm_model(
                        X_train=X_tr_seq,
                        y_train=y_tr,
                        seed=t_seed,
                        num_classes=3,
                        epochs=20,
                        batch_size=128,
                        lr=1e-3,
                        hidden_size=64,
                        num_layers=2,
                        dropout=0.2,
                    )
                
                # Within-Regime Evaluation: balanced test set
                y_te_bal = y_all[test_index]

                if experiment_model == 'LSTM':
                    X_te_bal = X_full[test_index]
                    y_pred_bal = predict_lstm_model(model, X_te_bal)
                else:
                    X_te_bal = X_full_flat[test_index]
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
                        if pre_para.para_type == 'volatility':
                            eval_threshold = int(eval_col_name.replace(label_suffix, "")) / 10.0
                        elif pre_para.para_type == 'horizon':
                            eval_threshold = int(eval_col_name.replace(label_suffix, ""))

                        if eval_mode == 'balanced':
                            eval_test_idx  = experiment_datsets[eval_col_name]["balanced_test_idx"]
                        elif eval_mode == 'raw':
                            eval_test_idx = test_idx
                        y_te_bal = y_all[eval_test_idx]

                        if experiment_model == 'LSTM':
                            X_te_bal = X_full[eval_test_idx]
                            y_pred_cross = predict_lstm_model(model, X_te_bal)
                        else:
                            X_te_bal = X_full_flat[eval_test_idx]
                            y_pred_cross = model.predict(X_te_bal)

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
                #financial return backtest

                if experiment_model == 'LSTM':
                    X_test = X_full[test_idx]
                    y_test_pred = predict_lstm_model(model, X_test)
                else:
                    X_test = X_full_flat[test_idx]
                    y_test_pred = model.predict(X_test)
                begin_time = time.time()
                financial_metrics = compute_financial_returns(
                    df=df,
                    pre_para = pre_para,
                    para_horizon = para_horizon,
                    window_indices=window_indices,
                    sample_idx=test_idx,
                    y_pred=y_test_pred,
                    label_col=col_name,
                    initial_cash=10000.0,
                    fee_rates=[0.0, 0.0005],
                    price_col="close",
                )
                end_time = time.time()
                logger.info(f"compute_financial_returns takes time: {(end_time - begin_time):.2f} seconds")
                for fm in financial_metrics:
                    financial_metrics_list.append(
                        {
                            "run_id": run_id,
                            "model": experiment_model,
                            "eval_mode": "raw_test",
                            "label_name": col_name,
                            "threshold": threshold,
                            "test_size": int(len(test_idx)),
                            **fm,
                        }
                    )

    self_eval_df = pd.DataFrame(self_eval_rows)
    cross_eval_df = pd.DataFrame(cross_eval_rows)
    financial_return_df = pd.DataFrame(financial_metrics_list)

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

    financial_summary_df = (
            financial_return_df
            .groupby(["model", "fee_rate", "threshold"])
            .agg(
                n_runs=("run_id", "nunique"),
                test_size=("test_size", "first"),
                round_trip_fee=("round_trip_fee", "first"),

                # strategy-level
                strategy_total_return_mean=("strategy_total_return", "mean"),
                strategy_total_return_std=("strategy_total_return", "std"),
                strategy_total_pnl_mean=("strategy_total_pnl", "mean"),
                strategy_total_pnl_std=("strategy_total_pnl", "std"),
                final_cash_mean=("final_cash", "mean"),
                final_cash_std=("final_cash", "std"),
                strategy_trade_count_mean=("strategy_trade_count", "mean"),
                strategy_trade_count_std=("strategy_trade_count", "std"),
                strategy_avg_trade_return_mean=("strategy_avg_trade_return", "mean"),
                strategy_avg_trade_return_std=("strategy_avg_trade_return", "std"),
                strategy_win_rate_mean=("strategy_win_rate", "mean"),
                strategy_win_rate_std=("strategy_win_rate", "std"),
                strategy_skipped_overlap_mean=("strategy_skipped_overlap", "mean"),
                strategy_skipped_overlap_std=("strategy_skipped_overlap", "std"),

                # signal-level
                signal_avg_return_mean=("signal_avg_return", "mean"),
                signal_avg_return_std=("signal_avg_return", "std"),
                signal_total_pnl_mean=("signal_total_pnl", "mean"),
                signal_total_pnl_std=("signal_total_pnl", "std"),
                signal_avg_pnl_mean=("signal_avg_pnl", "mean"),
                signal_avg_pnl_std=("signal_avg_pnl", "std"),
                signal_win_rate_mean=("signal_win_rate", "mean"),
                signal_win_rate_std=("signal_win_rate", "std"),
                signal_trade_count_mean=("signal_trade_count", "mean"),
                signal_trade_count_std=("signal_trade_count", "std"),

                # long / short breakdown
                signal_long_count_mean=("signal_long_count", "mean"),
                signal_long_count_std=("signal_long_count", "std"),
                signal_short_count_mean=("signal_short_count", "mean"),
                signal_short_count_std=("signal_short_count", "std"),
                signal_long_avg_return_mean=("signal_long_avg_return", "mean"),
                signal_long_avg_return_std=("signal_long_avg_return", "std"),
                signal_short_avg_return_mean=("signal_short_avg_return", "mean"),
                signal_short_avg_return_std=("signal_short_avg_return", "std"),
            )
            .reset_index()
            .sort_values(["model", "fee_rate", "threshold"])
        )

    self_eval_df.to_csv(os.path.join(save_dir, "self_eval_all_runs.csv"), index=False)
    cross_eval_df.to_csv(os.path.join(save_dir, "cross_eval_all_runs.csv"), index=False)
    self_summary_df.to_csv(os.path.join(save_dir, "self_eval_summary_mean_std.csv"), index=False)
    cross_train_summary_df.to_csv(os.path.join(save_dir, "cross_train_summary_mean_std.csv"), index=False)
    cross_train_batch_summary_df.to_csv(os.path.join(save_dir, "cross_train_batch_summary_mean_std.csv"), index=False)
    cross_eval_summary_df.to_csv(os.path.join(save_dir, "cross_eval_summary_std.csv"), index=False)
    cross_eval_batch_summary_df.to_csv(os.path.join(save_dir, "cross_eval_batch_summary_mean_std.csv"), index=False)

    financial_return_df.to_csv(
        os.path.join(save_dir, "financial_return_all_runs.csv"),
        index=False,
    )

    financial_summary_df.to_csv(
        os.path.join(save_dir, "financial_return_summary_mean_std.csv"),
        index=False,
    )

    plot_self_eval_mean_std(self_summary_df, save_dir, metric="macro_f1", para_type = pre_para.para_type)
    plot_self_eval_mean_std(self_summary_df, save_dir, metric="balanced_accuracy",para_type = pre_para.para_type)
    plot_self_eval_mean_std(self_summary_df, save_dir, metric="mcc",para_type = pre_para.para_type)

    plot_cross_train_batch_summary(cross_train_batch_summary_df, save_dir, metric="macro_f1", total_runs=total_runs,para_type = pre_para.para_type)
    plot_cross_train_batch_summary(cross_train_batch_summary_df, save_dir, metric="mcc", total_runs=total_runs,para_type = pre_para.para_type)
    plot_cross_eval_batch_summary(cross_eval_batch_summary_df, save_dir, metric="macro_f1", total_runs=total_runs,para_type = pre_para.para_type)
    plot_cross_eval_batch_summary(cross_eval_batch_summary_df, save_dir, metric="mcc", total_runs=total_runs,para_type = pre_para.para_type)
    plot_train_vs_eval_summary(cross_train_batch_summary_df, cross_eval_batch_summary_df, save_dir, metric="macro_f1", total_runs=total_runs,para_type = pre_para.para_type)
    plot_train_vs_eval_summary(cross_train_batch_summary_df, cross_eval_batch_summary_df, save_dir, metric="mcc", total_runs=total_runs,para_type = pre_para.para_type)
    plot_cross_eval_heatmap(cross_matrix_summary_df, save_dir, metric="macro_f1",annotate_std=True,para_type = pre_para.para_type)
    plot_cross_eval_heatmap(cross_matrix_summary_df, save_dir, metric="mcc",annotate_std=True,para_type = pre_para.para_type)

    plot_financial_return_mean_std(
        financial_summary_df,
        save_dir,
        metric="strategy_total_return",
        para_type = pre_para.para_type,
    )

    plot_financial_return_mean_std(
        financial_summary_df,
        save_dir,
        metric="signal_avg_return",
        para_type = pre_para.para_type,
    )
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