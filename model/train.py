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
  5) Randomize per-run model training order to spread out LSTM GPU usage across processes
  6) Free GPU memory once an entire LSTM stage (all label_cols) finishes for a run
  7) Dynamically probe current GPU memory usage before an LSTM stage and fall back
     to CPU training if there isn't enough headroom, instead of a hardcoded concurrency cap
"""

import os,sys,re,json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
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
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import ConvergenceWarning
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import copy, shutil

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
sys.path.append(current_work_dir)  # ensure plot_results.py (same dir) is importable regardless of CWD
from data_process import common
from model.data_loader import TimeSeriesWindowDataset
from model.train_config import *
from model.models import lstm
from analysis import plot_results  # decoupled plotting module: reads persisted summary CSVs / writes figures
from analysis import compose_report_images

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

def generate_random_predictions(n: int, seed: int, p: np.ndarray = None) -> np.ndarray:
    """
    Random-guess baseline predictions.

    p=None -> uniform over the 3 classes (1/3 each).
    p=[p_pos, p_neg, p_neu] -> sample according to given class probabilities.
    """
    rng = np.random.default_rng(seed)
    return rng.choice(
        [common.Signal.POSITIVE, common.Signal.NEGATIVE, common.Signal.NEUTRAL],
        size=int(n),
        replace=True,
        p=p,
    ).astype(np.int64)


def compute_class_proportions(y: np.ndarray) -> np.ndarray:
    """
    Empirical class proportions [p_pos, p_neg, p_neu] of `y`, in the same
    class order used by generate_random_predictions. Falls back to uniform
    if `y` is empty (shouldn't normally happen).
    """
    classes = [common.Signal.POSITIVE, common.Signal.NEGATIVE, common.Signal.NEUTRAL]
    counts = np.array([np.sum(y == c) for c in classes], dtype=float)
    total = counts.sum()
    if total <= 0:
        return np.full(len(classes), 1.0 / len(classes))
    return counts / total


def prepare_parameter_regime_datasets(
    logger: logging.Logger, seed, labels_matrix,
    train_idx, val_idx, test_idx, label_cols,
    strictest_train_target_n, strictest_val_target_n, strictest_test_target_n,
):
    experiment_datsets: dict[str, dict] = {}
    train_target_n = strictest_train_target_n
    val_target_n = strictest_val_target_n
    test_target_n = strictest_test_target_n
    logger.warning(
        "strictest label column is %s, with train target n=%d, val target n=%d, test target n=%d",
        label_cols[-1], train_target_n, val_target_n, test_target_n,
    )

    for col_idx, col_name in enumerate(label_cols):
        y_all = labels_matrix[:, col_idx].astype(np.int64)

        balanced_train_idx = sample_balanced_indices_downsample(
            logger, y=y_all, candidate_idx=train_idx,
            size=train_target_n, seed=seed + 10000 + col_idx * 100,
        )
        balanced_val_idx = sample_balanced_indices_downsample(
            logger, y=y_all, candidate_idx=val_idx,
            size=val_target_n, seed=seed + 10000 + col_idx * 100 + 2,
        )
        balanced_test_idx = sample_balanced_indices_downsample(
            logger, y=y_all, candidate_idx=test_idx,
            size=test_target_n, seed=seed + 10000 + col_idx * 100 + 1,
        )
        experiment_datsets[col_name] = {
            "balanced_train_idx": balanced_train_idx,
            "balanced_val_idx": balanced_val_idx,
            "balanced_test_idx": balanced_test_idx,
        }
    return experiment_datsets


# -----------------------------
# GPU memory probing helpers
# -----------------------------

def _get_own_gpu_process_memory_usage(device_index: int = 0) -> list[float]:
    """
    Query the GPU's compute-process list and return each process's memory
    usage in MiB.

    Returns an empty list when:
      - pynvml is not installed
      - NVML query fails for any reason (no permission, driver issue, etc.)
      - there are currently no compute processes on this GPU

    Callers should treat an empty list as "no evidence of other usage",
    which under this experiment's fixed-shape assumption means it's safe
    to proceed with GPU training.
    """
    if not _PYNVML_AVAILABLE:
        return []
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        usage_mib = [
            p.usedGpuMemory / (1024 ** 2)
            for p in procs
            if p.usedGpuMemory is not None
        ]
        return usage_mib
    except Exception:
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

def _get_free_gpu_memory_mib(device_index: int = 0):
    if not _PYNVML_AVAILABLE:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.free / (1024 ** 2)
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

def _select_device_for_lstm(
    logger: logging.Logger,
    safety_margin_mib: float = 300.0,
    device_index: int = 0,
    max_gpu_processes: int = 10,
) -> torch.device:
    """
    Decide whether this run's upcoming LSTM stage (which will sequentially
    train every label_col with an identical shape/config) should run on GPU
    or fall back to CPU.
 
    Two independent gates, both must pass for GPU to be used:
 
    1) Concurrency cap (`max_gpu_processes`):
       Even when there is plenty of free memory, too many processes training
       on the GPU at the same time slows every one of them down (kernel
       launch contention, SM oversubscription, etc.). If the GPU already has
       `max_gpu_processes` or more compute processes running, this run falls
       back to CPU regardless of how much free memory remains.
 
    2) Memory estimation:
       Every LSTM training run in this experiment has an identical shape
       (train_target_n, batch_size, hidden_size, ... are all fixed), so any
       currently-running process's GPU memory usage is a good proxy for what
       THIS run will need: required memory ~= max(usage across all current
       GPU processes) + margin.
 
    If no processes are currently using the GPU (query returns empty, either
    because the GPU is idle or because the query itself is unavailable/
    failed), there is nothing to conflict with on either gate, so GPU is
    used directly.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
 
    try:
        free_mib = _get_free_gpu_memory_mib(device_index)

        if free_mib is None:
            logger.warning(
                "Failed to query GPU free memory, falling back to CPU"
            )
            return torch.device("cpu")
    except Exception:
        logger.exception("Failed to query GPU free memory, falling back to CPU")
        return torch.device("cpu")
 
    proc_usages = _get_own_gpu_process_memory_usage(device_index)
 
    if not proc_usages:
        # No processes detected (idle GPU, or query unavailable/failed):
        # nothing to compete with on either gate, so allow GPU directly.
        logger.info(
            f"No GPU process usage detected (free={free_mib:.0f}MiB), "
            f"proceeding with GPU for this LSTM stage"
        )
        return torch.device(f"cuda:{device_index}")
 
    # ---- Gate 1: concurrency cap ----
    current_process_count = len(proc_usages)
    # if current_process_count >= max_gpu_processes:
    #     logger.warning(
    #         f"GPU already has {current_process_count} compute processes "
    #         f"(>= max_gpu_processes={max_gpu_processes}), this LSTM stage "
    #         f"will run on CPU instead to avoid overcrowding the GPU"
    #     )
    #     return torch.device("cpu")
    # else:
    #     logger.info(
    #         f"GPU already has {current_process_count} compute processes "
    #     )
 
    # ---- Gate 2: memory estimation ----
    estimated_required_mib = max(proc_usages) + safety_margin_mib
    logger.info(
        f"GPU processes currently: {current_process_count} "
        f"(max_gpu_processes={max_gpu_processes}), "
        f"max usage={max(proc_usages):.0f}MiB, free={free_mib:.0f}MiB, "
        f"estimated required for this run={estimated_required_mib:.0f}MiB"
    )
 
    if free_mib < estimated_required_mib:
        logger.warning(
            f"GPU free memory low ({free_mib:.0f}MiB < estimated required "
            f"{estimated_required_mib:.0f}MiB), this LSTM stage will run on CPU instead"
        )
        return torch.device("cpu")
 
    return torch.device(f"cuda:{device_index}")


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    num_classes: int = 3,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    patience: int = 5,
    min_delta: float = 1e-4,
    logger: logging.Logger = None,
    device: torch.device = None,
):
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

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

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_ds)

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_pred = torch.argmax(val_logits, dim=1).cpu().numpy()
        val_f1 = f1_score(y_val, val_pred, average="macro")

        if logger is not None:
            logger.info(
                f"[LSTM] epoch={epoch+1}/{epochs} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}"
            )

        # ---- early stopping on val_loss ----
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if logger is not None:
                    logger.info(f"[LSTM] early stopping at epoch={epoch+1}, best_epoch={best_epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

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

def _get_worker_logger(run_id: int, log_dir: str) -> logging.Logger:
    logger = logging.getLogger(f"run_worker_{run_id}")
    if not logger.handlers:
        os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(log_dir, f"run_{run_id}.log"))
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def run_single_iteration(
    run_id: int,
    total_runs: int,
    train_cfg_seed: int,
    labels_matrix: np.ndarray,
    label_cols: list,
    label_suffix: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    window_indices: np.ndarray,
    X_full: np.ndarray,
    X_full_flat: np.ndarray,
    df: pd.DataFrame,
    pre_para: common.BaseDefine,
    train_cfg: TrainConfig,
    strictest_train_target_n: int,
    strictest_val_target_n: int,
    strictest_test_target_n: int,
    log_dir: str,
) -> Tuple[list, list, list]:
    """
    单个 run_id 的完整训练+评估逻辑，在独立进程中执行。
    返回该 run 产生的三份行数据，供主进程汇总。
    """
    # 避免 20 个进程同时抢占 CPU 线程（sklearn / numpy BLAS）
    torch.set_num_threads(1)

    logger = _get_worker_logger(run_id, log_dir)
    logger.info(f"Starting run {run_id+1}/{total_runs}")

    self_eval_rows_local = []
    cross_eval_rows_local = []
    financial_metrics_local = []

    t_seed = train_cfg_seed + run_id * 10000

    experiment_datsets = prepare_parameter_regime_datasets(
        logger, t_seed, labels_matrix,
        train_idx, val_idx, test_idx, label_cols,
        strictest_train_target_n, strictest_val_target_n, strictest_test_target_n,
    )
    random_baseline_done = set()

    # 用独立的随机序列打乱模型训练顺序，避免所有进程的 LSTM 训练同时撞在一起。
    # 用与 t_seed 不同的偏移量，保证这里的随机性和数据采样的随机性互不干扰。
    model_order = ['DecisionTree', 'LogisticRegression', 'LSTM']
    order_rng = np.random.default_rng(t_seed + 777777)
    order_rng.shuffle(model_order)
    logger.info(f"run_id={run_id} model training order: {model_order}")

    for experiment_model in model_order:
        if 'd' in pre_para.interval and experiment_model == 'LSTM':
            logger.info(
                f"skip LSTM on {pre_para.symbol}_{pre_para.interval}, "
                f"since the dataset is insufficient to complete the training."
            )
            continue

        # 在整个 LSTM 阶段（会连续训练本 run_id 下所有 label_cols）开始前，
        # 只查询一次 GPU 显存占用情况，据此决定这一整段是走 GPU 还是回退 CPU。
        # 由于本实验每次 LSTM 训练的 shape 完全固定，当前 GPU 上任意一个进程
        # 的显存占用都可以作为本次训练所需显存的预估；若没有查到任何进程占用
        # （GPU 空闲，或查询本身不可用/失败），则直接允许使用 GPU。
        lstm_device = None
        if experiment_model == 'LSTM':
            lstm_device = _select_device_for_lstm(logger)
            logger.info(f"run_id={run_id} LSTM stage will use device={lstm_device}")

        for col_idx, col_name in enumerate(label_cols):
            logger.info(f"{experiment_model} train {col_name}")
            y_all = labels_matrix[:, col_idx].astype(np.int64)

            if pre_para.para_type == 'volatility':
                threshold = int(col_name.replace(label_suffix, "")) / 10.0
                para_horizon = pre_para.predict_num
            elif pre_para.para_type == 'horizon':
                threshold = int(col_name.replace(label_suffix, ""))
                para_horizon = threshold

            train_index = experiment_datsets[col_name]["balanced_train_idx"]
            n_eff = len(train_index) // 3
            test_index = experiment_datsets[col_name]["balanced_test_idx"]

            X_tr_flat = X_full_flat[train_index]
            X_tr_seq = X_full[train_index]
            y_tr = y_all[train_index]

            logger.info(
                f"Starting experiment: Model={experiment_model} | "
                f"train_size={len(train_index)} (effective balanced size per class={n_eff}) | "
                f"test_size={len(test_index)}"
            )

            if experiment_model == 'LogisticRegression':
                model = LogisticRegression(
                    solver=train_cfg.lr_solver,
                    max_iter=train_cfg.lr_max_iter,
                    C=train_cfg.lr_C,
                )
                model.fit(X_tr_flat, y_tr)
            elif experiment_model == 'DecisionTree':
                model = DecisionTreeClassifier(
                    min_samples_leaf=max(20, int(0.01 * len(y_tr))),
                    max_depth=5,
                    random_state=t_seed,
                )
                model.fit(X_tr_flat, y_tr)
            elif experiment_model == 'LSTM':
                val_index = experiment_datsets[col_name]["balanced_val_idx"]
                X_val_seq = X_full[val_index]
                y_val = y_all[val_index]
                model = train_lstm_model(
                    X_train=X_tr_seq,
                    y_train=y_tr,
                    X_val=X_val_seq,
                    y_val=y_val,
                    seed=t_seed,
                    num_classes=3,
                    epochs=20,
                    batch_size=128,
                    lr=1e-3,
                    hidden_size=64,
                    num_layers=2,
                    dropout=0.2,
                    patience=5,
                    logger=logger,
                    device=lstm_device,
                )

            # Within-Regime Evaluation (balanced test set + original/raw test set)
            class_labels = [common.Signal.POSITIVE, common.Signal.NEGATIVE, common.Signal.NEUTRAL]

            n_pos = int(np.sum(y_all[test_idx] == common.Signal.POSITIVE))
            n_neg = int(np.sum(y_all[test_idx] == common.Signal.NEGATIVE))
            n_neu = int(np.sum(y_all[test_idx] == common.Signal.NEUTRAL))
            balanced_class_size = min(n_pos, n_neg, n_neu)

            logger.info(
                f"{col_name} | raw test counts: pos={n_pos}, neg={n_neg}, neu={n_neu}, "
                f"balanced_class_size={balanced_class_size}"
            )

            # ---------------------------------------------------------------
            # RandomPredict baseline: computed once per label (col_name),
            # regardless of experiment_model, since a random predictor does
            # not depend on any trained model. Both balanced and raw variants
            # sample according to the TRUE class proportions of their own
            # eval set (p=None would be uniform): for the balanced test set
            # this is ~1/3 each by construction, for the raw test set this
            # reflects the real class imbalance, so a real model's macro_f1
            # actually has to beat "just guessing the base rate" to look
            # good, not merely beat a naive uniform guess. The SAME raw-test
            # random predictions are reused for both the self-eval baseline
            # below and the financial-return baseline further down, so the
            # two baselines are consistent with each other.
            # ---------------------------------------------------------------
            if col_name not in random_baseline_done:
                p_bal = compute_class_proportions(y_all[test_index])
                p_raw = compute_class_proportions(y_all[test_idx])

                y_random_pred_bal = generate_random_predictions(
                    n=len(test_index), seed=t_seed + 900000 + col_idx, p=p_bal,
                )
                y_random_pred_raw = generate_random_predictions(
                    n=len(test_idx), seed=t_seed + 900000 + col_idx + 500000, p=p_raw,
                )

                for rb_eval_mode, rb_eval_idx, rb_y_pred in [
                    ('balanced', test_index, y_random_pred_bal),
                    ('raw', test_idx, y_random_pred_raw),
                ]:
                    y_te_rb = y_all[rb_eval_idx]
                    f1_rb = float(f1_score(y_te_rb, rb_y_pred, average="macro"))
                    mcc_rb = float(matthews_corrcoef(y_te_rb, rb_y_pred))
                    acc_rb = float(accuracy_score(y_te_rb, rb_y_pred))
                    p_rb, r_rb, f_rb, _ = precision_recall_fscore_support(
                        y_te_rb, rb_y_pred, labels=class_labels, zero_division=0
                    )

                    self_eval_rows_local.append(
                        {
                            "run_id": run_id,
                            "model": "RandomPredict",
                            "eval_mode": rb_eval_mode,
                            "label_name": col_name,
                            "threshold": threshold,
                            "macro_f1": f1_rb,
                            "mcc": mcc_rb,
                            "accuracy": acc_rb,
                            "p_pos": p_rb[0], "r_pos": r_rb[0], "f_pos": f_rb[0],
                            "p_neg": p_rb[1], "r_neg": r_rb[1], "f_neg": f_rb[1],
                            "p_neu": p_rb[2], "r_neu": r_rb[2], "f_neu": f_rb[2],
                            "n_eff": n_eff,
                            "train_size": 0,
                            "test_size": int(len(y_te_rb)),
                            "test_pos_raw": n_pos,
                            "test_neg_raw": n_neg,
                            "test_neu_raw": n_neu,
                            "balanced_class_size": balanced_class_size,
                        }
                    )

                logger.info(
                    f"{col_name} | RandomPredict self-eval baseline computed (balanced + raw); "
                    f"p_bal(pos,neg,neu)={np.round(p_bal, 3).tolist()}, "
                    f"p_raw(pos,neg,neu)={np.round(p_raw, 3).tolist()}"
                )

                # Reuse the raw-test random predictions computed above for the financial baseline.
                begin_time = time.time()
                random_financial_metrics = compute_financial_returns(
                    df=df, pre_para=pre_para, para_horizon=para_horizon,
                    window_indices=window_indices, sample_idx=test_idx,
                    y_pred=y_random_pred_raw, label_col=col_name,
                    initial_cash=10000.0, fee_rates=[0.0, 0.0005], price_col="close",
                )
                end_time = time.time()
                logger.info(
                    f"compute RandomPredict financial_returns for {col_name} "
                    f"takes time: {(end_time - begin_time):.2f} seconds"
                )
                for fm in random_financial_metrics:
                    financial_metrics_local.append(
                        {
                            "run_id": run_id, "model": "RandomPredict", "eval_mode": "raw_test",
                            "label_name": col_name, "threshold": threshold,
                            "test_size": int(len(test_idx)), **fm,
                        }
                    )
                random_baseline_done.add(col_name)

            y_pred_raw = None  # captured below when self_eval_mode == 'raw'; reused later for financial backtest

            for self_eval_mode in ['balanced', 'raw']:
                eval_idx = test_index if self_eval_mode == 'balanced' else test_idx

                y_te = y_all[eval_idx]
                if experiment_model == 'LSTM':
                    X_te = X_full[eval_idx]
                    y_pred = predict_lstm_model(model, X_te)
                else:
                    X_te = X_full_flat[eval_idx]
                    y_pred = model.predict(X_te)

                if self_eval_mode == 'raw':
                    y_pred_raw = y_pred  # reused later for financial backtest, avoids recomputation

                f1_ = float(f1_score(y_te, y_pred, average="macro"))
                mcc_ = float(matthews_corrcoef(y_te, y_pred))
                acc_ = float(accuracy_score(y_te, y_pred))
                p_class, r_class, f_class, _ = precision_recall_fscore_support(
                    y_te, y_pred, labels=class_labels, zero_division=0
                )

                logger.info(
                    f"{col_name} | {experiment_model} self-eval [{self_eval_mode}]: "
                    f"macro_f1={f1_:.4f}, mcc={mcc_:.4f}, acc={acc_:.4f}"
                )

                self_eval_rows_local.append(
                    {
                        "run_id": run_id,
                        "model": experiment_model,
                        "eval_mode": self_eval_mode,
                        "label_name": col_name,
                        "threshold": threshold,
                        "macro_f1": f1_,
                        "mcc": mcc_,
                        "accuracy": acc_,
                        "p_pos": p_class[0], "r_pos": r_class[0], "f_pos": f_class[0],
                        "p_neg": p_class[1], "r_neg": r_class[1], "f_neg": f_class[1],
                        "p_neu": p_class[2], "r_neu": r_class[2], "f_neu": f_class[2],
                        "n_eff": n_eff,
                        "train_size": int(3 * n_eff),
                        "test_size": int(len(y_te)),
                        "test_pos_raw": n_pos,
                        "test_neg_raw": n_neg,
                        "test_neu_raw": n_neu,
                        "balanced_class_size": balanced_class_size,
                    }
                )

            # Cross-Regime Evaluation
            for eval_mode in ['balanced', 'raw']:
                for eval_col_idx, eval_col_name in enumerate(label_cols):
                    y_all_eval = labels_matrix[:, eval_col_idx].astype(np.int64)
                    if pre_para.para_type == 'volatility':
                        eval_threshold = int(eval_col_name.replace(label_suffix, "")) / 10.0
                    elif pre_para.para_type == 'horizon':
                        eval_threshold = int(eval_col_name.replace(label_suffix, ""))

                    if eval_mode == 'balanced':
                        eval_test_idx = experiment_datsets[eval_col_name]["balanced_test_idx"]
                    elif eval_mode == 'raw':
                        eval_test_idx = test_idx
                    y_te_cross = y_all_eval[eval_test_idx]

                    if experiment_model == 'LSTM':
                        X_te_cross = X_full[eval_test_idx]
                        y_pred_cross = predict_lstm_model(model, X_te_cross)
                    else:
                        X_te_cross = X_full_flat[eval_test_idx]
                        y_pred_cross = model.predict(X_te_cross)

                    f1_cross = float(f1_score(y_te_cross, y_pred_cross, average="macro"))
                    mcc_cross = float(matthews_corrcoef(y_te_cross, y_pred_cross))
                    acc_cross = float(accuracy_score(y_te_cross, y_pred_cross))
                    p_class_cross, r_class_cross, f_class_cross, _ = precision_recall_fscore_support(
                        y_te_cross, y_pred_cross, labels=class_labels, zero_division=0
                    )

                    cross_eval_rows_local.append(
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
                            "mcc": mcc_cross,
                            "accuracy": acc_cross,
                            "p_pos": p_class_cross[0], "r_pos": r_class_cross[0], "f_pos": f_class_cross[0],
                            "p_neg": p_class_cross[1], "r_neg": r_class_cross[1], "f_neg": f_class_cross[1],
                            "p_neu": p_class_cross[2], "r_neu": r_class_cross[2], "f_neu": f_class_cross[2],
                            "test_size": int(len(y_te_cross)),
                        }
                    )

            # Financial return backtest (RandomPredict baseline already handled once per label above)
            y_test_pred = y_pred_raw

            begin_time = time.time()
            financial_metrics = compute_financial_returns(
                df=df, pre_para=pre_para, para_horizon=para_horizon,
                window_indices=window_indices, sample_idx=test_idx,
                y_pred=y_test_pred, label_col=col_name,
                initial_cash=10000.0, fee_rates=[0.0, 0.0005], price_col="close",
            )
            end_time = time.time()
            logger.info(f"compute_financial_returns takes time: {(end_time - begin_time):.2f} seconds")
            for fm in financial_metrics:
                financial_metrics_local.append(
                    {
                        "run_id": run_id, "model": experiment_model, "eval_mode": "raw_test",
                        "label_name": col_name, "threshold": threshold,
                        "test_size": int(len(test_idx)), **fm,
                    }
                )
        if experiment_model == 'LSTM':
            del model
            torch.cuda.empty_cache()

    logger.info(f"Finished run {run_id+1}/{total_runs}")
    return self_eval_rows_local, cross_eval_rows_local, financial_metrics_local

def run_fixed_neutral_subsampling_experiment(
    logger: logging.Logger,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    pre_para: common.BaseDefine,
    prep_output_dir: str,
    save_dir: str,
    max_workers: int ,   # None -> 默认用 total_runs（你说显存够20个并发）
):
    set_seed(train_cfg.seed)

    df = common.load_train_df_from_dir(prep_output_dir)
    if pre_para is None:
        pre_para = common.load_pre_params_from_dir(prep_output_dir)
        logger.info(f"load pre_para from {prep_output_dir}")
    save_dir = os.path.join(save_dir, f"{pre_para.symbol}_{pre_para.interval}", pre_para.label_type, pre_para.para_type)
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
        df=df, kline_interval_ms=kline_interval_ms,
        feature_cols=train_cfg.feature_conf_list, label_col=label_cols[0],
        window=train_cfg.seq_len, stride=train_cfg.stride,
        use_cache=False, show_feature_distribution=True,
    )

    window_indices = master_ds.indices
    labels_matrix = df.loc[window_indices, label_cols].values.astype(np.int64)

    X_full = master_ds.X.detach().cpu().numpy()
    logger.info(f"X_full shape: {X_full.shape} | labels_matrix shape: {labels_matrix.shape}")
    M = X_full.shape[0]
    X_full_flat = X_full.reshape(M, -1)

    tr_rng, va_rng, te_rng = chrono_split_by_window_ends(M, data_cfg.train_ratio, data_cfg.val_ratio)
    train_idx = np.arange(tr_rng[0], tr_rng[1])
    val_idx = np.arange(va_rng[0], va_rng[1])
    test_idx = np.arange(te_rng[0], te_rng[1])

    logger.info(f"Master windows M={M} | train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")

    total_runs = 50

    strictest_train_target_n = np.inf
    strictest_val_target_n = np.inf
    strictest_test_target_n = np.inf
    for col_idx, col_name in enumerate(label_cols):
        y_all = labels_matrix[:, col_idx].astype(np.int64)
        _, _, _, train_target_n = compute_minimum_size(y_all, train_idx)
        _, _, _, val_target_n = compute_minimum_size(y_all, val_idx)
        _, _, _, test_target_n = compute_minimum_size(y_all, test_idx)
        if train_target_n < strictest_train_target_n:
            strictest_train_target_n = train_target_n
            logger.info(f"New strictest train target n={strictest_train_target_n} found at column {col_name}")
        if val_target_n < strictest_val_target_n:
            strictest_val_target_n = val_target_n
            logger.info(f"New strictest val target n={strictest_val_target_n} found at column {col_name}")
        if test_target_n < strictest_test_target_n:
            strictest_test_target_n = test_target_n
            logger.info(f"New strictest test target n={strictest_test_target_n} found at column {col_name}")

    # =====================================================
    # 并发跑 total_runs 个 run_id
    # =====================================================
    log_dir = os.path.join(save_dir, "run_logs")
    os.makedirs(log_dir, exist_ok=True)

    self_eval_rows, cross_eval_rows, financial_metrics_list = [], [], []

    ctx = mp.get_context("spawn")  # 必须用 spawn，避免 CUDA + fork 的问题

    logger.info(f"Launching {total_runs} runs with max_workers={max_workers} (spawn context)")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(
                run_single_iteration,
                run_id=run_id,
                total_runs=total_runs,
                train_cfg_seed=train_cfg.seed,
                labels_matrix=labels_matrix,
                label_cols=label_cols,
                label_suffix=label_suffix,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                window_indices=window_indices,
                X_full=X_full,
                X_full_flat=X_full_flat,
                df=df,
                pre_para=pre_para,
                train_cfg=train_cfg,
                strictest_train_target_n=strictest_train_target_n,
                strictest_val_target_n=strictest_val_target_n,
                strictest_test_target_n=strictest_test_target_n,
                log_dir=log_dir,
            ): run_id
            for run_id in range(total_runs)
        }

        completed = 0
        for future in as_completed(futures):
            run_id = futures[future]
            try:
                se_rows, ce_rows, fm_rows = future.result()
                self_eval_rows.extend(se_rows)
                cross_eval_rows.extend(ce_rows)
                financial_metrics_list.extend(fm_rows)
                completed += 1
                logger.info(
                    f"[{completed}/{total_runs}] run_id={run_id} finished, "
                    f"{len(se_rows)} self-eval rows collected"
                )
            except Exception:
                logger.exception(f"run_id={run_id} raised an exception")
                raise  # 任何一个 run 出错都直接终止，避免拿到不完整/有偏的结果

    # ---- 为了让保存的 CSV / 后续分析结果与运行顺序无关，按 run_id 排序 ----
    self_eval_rows.sort(key=lambda r: (r["run_id"], r["model"], r["threshold"]))
    cross_eval_rows.sort(key=lambda r: (r["run_id"], r["model"], r["train_threshold"], r["eval_threshold"]))
    financial_metrics_list.sort(key=lambda r: (r["run_id"], r["model"], r["threshold"], r["fee_rate"]))

    self_eval_df = pd.DataFrame(self_eval_rows)
    cross_eval_df = pd.DataFrame(cross_eval_rows)
    financial_return_df = pd.DataFrame(financial_metrics_list)

    self_summary_df = (self_eval_df.groupby(["model", "eval_mode", "threshold"]).agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            mcc_mean=("mcc", "mean"),
            mcc_std=("mcc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            p_pos_mean=("p_pos", "mean"), p_pos_std=("p_pos", "std"),
            r_pos_mean=("r_pos", "mean"), r_pos_std=("r_pos", "std"),
            f_pos_mean=("f_pos", "mean"), f_pos_std=("f_pos", "std"),
            p_neg_mean=("p_neg", "mean"), p_neg_std=("p_neg", "std"),
            r_neg_mean=("r_neg", "mean"), r_neg_std=("r_neg", "std"),
            f_neg_mean=("f_neg", "mean"), f_neg_std=("f_neg", "std"),
            n_runs=("run_id", "nunique"),
            train_size=("train_size", "first"),
            test_size=("test_size", "first"),
        )
        .reset_index()
    )

    cross_train_summary_df = (cross_eval_df.groupby(["run_id", "model", "eval_mode", "train_threshold"]).agg(
            macro_f1_mean=("macro_f1", "mean"),
            mcc_mean=("mcc", "mean"),
            accuracy_mean=("accuracy", "mean"),
            p_pos_mean=("p_pos", "mean"),
            r_pos_mean=("r_pos", "mean"),
            f_pos_mean=("f_pos", "mean"),
            p_neg_mean=("p_neg", "mean"),
            r_neg_mean=("r_neg", "mean"),
            f_neg_mean=("f_neg", "mean"),
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
            accuracy_mean_mean=("accuracy_mean", "mean"),
            accuracy_mean_std=("accuracy_mean", "std"),
            p_pos_mean_mean=("p_pos_mean", "mean"), p_pos_mean_std=("p_pos_mean", "std"),
            r_pos_mean_mean=("r_pos_mean", "mean"), r_pos_mean_std=("r_pos_mean", "std"),
            f_pos_mean_mean=("f_pos_mean", "mean"), f_pos_mean_std=("f_pos_mean", "std"),
            p_neg_mean_mean=("p_neg_mean", "mean"), p_neg_mean_std=("p_neg_mean", "std"),
            r_neg_mean_mean=("r_neg_mean", "mean"), r_neg_mean_std=("r_neg_mean", "std"),
            f_neg_mean_mean=("f_neg_mean", "mean"), f_neg_mean_std=("f_neg_mean", "std"),
            train_size=("train_size", "first"),
            test_size=("test_size", "first"),
        )
        .reset_index()
    )
    cross_eval_summary_df = (cross_eval_df.groupby(["run_id", "model", "eval_mode", "eval_threshold"]).agg(
            macro_f1_mean=("macro_f1", "mean"),
            mcc_mean=("mcc", "mean"),
            accuracy_mean=("accuracy", "mean"),
            p_pos_mean=("p_pos", "mean"),
            r_pos_mean=("r_pos", "mean"),
            f_pos_mean=("f_pos", "mean"),
            p_neg_mean=("p_neg", "mean"),
            r_neg_mean=("r_neg", "mean"),
            f_neg_mean=("f_neg", "mean"),
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
            accuracy_mean_mean=("accuracy_mean", "mean"),
            accuracy_mean_std=("accuracy_mean", "std"),
            p_pos_mean_mean=("p_pos_mean", "mean"), p_pos_mean_std=("p_pos_mean", "std"),
            r_pos_mean_mean=("r_pos_mean", "mean"), r_pos_mean_std=("r_pos_mean", "std"),
            f_pos_mean_mean=("f_pos_mean", "mean"), f_pos_mean_std=("f_pos_mean", "std"),
            p_neg_mean_mean=("p_neg_mean", "mean"), p_neg_mean_std=("p_neg_mean", "std"),
            r_neg_mean_mean=("r_neg_mean", "mean"), r_neg_mean_std=("r_neg_mean", "std"),
            f_neg_mean_mean=("f_neg_mean", "mean"), f_neg_mean_std=("f_neg_mean", "std"),
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
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
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

    # ---- Persist every summary table needed to (re)draw all figures later ----
    # This is the sole hand-off point to plot_results.py: as long as these
    # files exist under save_dir, plots can be regenerated at any time
    # without re-running any training/evaluation.
    self_eval_df.to_csv(os.path.join(save_dir, "self_eval_all_runs.csv"), index=False)
    cross_eval_df.to_csv(os.path.join(save_dir, "cross_eval_all_runs.csv"), index=False)
    self_summary_df.to_csv(os.path.join(save_dir, "self_eval_summary_mean_std.csv"), index=False)
    cross_train_summary_df.to_csv(os.path.join(save_dir, "cross_train_summary_mean_std.csv"), index=False)
    cross_train_batch_summary_df.to_csv(os.path.join(save_dir, "cross_train_batch_summary_mean_std.csv"), index=False)
    cross_eval_summary_df.to_csv(os.path.join(save_dir, "cross_eval_summary_std.csv"), index=False)
    cross_eval_batch_summary_df.to_csv(os.path.join(save_dir, "cross_eval_batch_summary_mean_std.csv"), index=False)
    cross_matrix_summary_df.to_csv(os.path.join(save_dir, "cross_matrix_summary_mean_std.csv"), index=False)

    financial_return_df.to_csv(
        os.path.join(save_dir, "financial_return_all_runs.csv"),
        index=False,
    )

    financial_summary_df.to_csv(
        os.path.join(save_dir, "financial_return_summary_mean_std.csv"),
        index=False,
    )

    meta = {
        "symbol": pre_para.symbol,
        "interval": pre_para.interval,
        "label_type": pre_para.label_type,
        "para_type": pre_para.para_type,
        "total_runs": total_runs,
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ---- Copy preparation.py's provenance files as-is into save_dir ----
    for fname in ("data_config_meta.json", "label_sweep_meta.json"):
        src = os.path.join(prep_output_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(save_dir, fname))
            logger.info(f"copied {src} -> {save_dir}")
        else:
            logger.warning(f"{src} not found, skipping copy")

        # ---- Plotting is fully decoupled: hand the in-memory summary tables to

    # ---- Plotting is fully decoupled: hand the in-memory summary tables to
    # plot_results.py so figures are produced right away. The exact same
    # CSVs are on disk under save_dir, so `python plot_results.py --save_dir
    # <save_dir>` can regenerate every figure later without re-training. ----
    logger.info(f"Training/evaluation done, generating plots via plot_results.py into {save_dir}")
    plot_results.generate_all_plots(
        save_dir=save_dir,
        para_type=pre_para.para_type,
        total_runs=total_runs,
        self_summary_df=self_summary_df,
        cross_train_batch_summary_df=cross_train_batch_summary_df,
        cross_eval_batch_summary_df=cross_eval_batch_summary_df,
        cross_matrix_summary_df=cross_matrix_summary_df,
        financial_summary_df=financial_summary_df,
    )
    compose_report_images.run_compose_report(pre_para)
# -----------------------------
# Entrypoint
# -----------------------------
def main(
    logger: logging.Logger,
    train_cfg: TrainConfig ,
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
        max_workers = 25
    )

if __name__ == "__main__":
    logger, _ = common.setup_session_logger(sub_folder="train", file_level=logging.DEBUG)

    begin_time = time.time()
    prep_output_dir = common.DATA_OUT_DIR

    cfg = TrainConfig()

    main(logger, train_cfg=cfg, pre_para=None, prep_output_dir=prep_output_dir)

    end_time = time.time()
    logger.info(f"Total time: {(end_time - begin_time):.2f} seconds")