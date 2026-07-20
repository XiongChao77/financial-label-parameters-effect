"""
turning_specificity.py
-----------------------
Simplified relative to earlier versions of this analysis: no permutation
test, no significance threshold. Just:

    1. Smooth each performance curve (Savitzky-Golay filter) to separate the
       genuine shape from point-to-point sampling noise.
    2. Find the curve's own local extrema (turning points: local minima /
       maxima) directly on the smoothed curve.
    3. List those turning points side by side with the label-proportion x
       Gaussian-reference crossing points (already computed and saved by
       LabelRatioCurveAnalyzer) for the same experimental group, so the two
       can be compared directly by inspection.

Whether a turning point sits "close enough" to a crossing point to be
considered meaningful is left to manual/visual judgement rather than a
formal statistical test.

A curve with NO detected turning point (monotonic after smoothing) is
reported as such (turning_point_count = 0) rather than forcing a comparison
-- this is a meaningful result, not a failure mode to work around.

Reuses the same path/loader conventions as crossing_specificity.py so this
module can sit alongside it and read the same upstream CSVs.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
sys.path.append(current_work_dir)
try:
    from data_process import common
except ModuleNotFoundError:
    # Core algorithm (find_turning_points) has no dependency on the project
    # package and can be imported/tested standalone; `common` is only
    # needed by the file-loading and batch-run functions further down,
    # which will raise clearly if actually called without it.
    common = None


# ------------------------------------------------------------------
# Path helpers (identical convention to crossing_specificity.py)
# ------------------------------------------------------------------

def crossings_dir(para, output_dir=None):
    output_dir = output_dir or common.OUTPUT_DIR
    return os.path.join(
        output_dir, "regime_discovery_output",
        f"{para.symbol}_{para.interval}", para.label_type, para.para_type,
    )


def train_dir(para, save_dir=None):
    save_dir = save_dir or common.TRAIN_OUT_DIR
    return os.path.join(save_dir, f"{para.symbol}_{para.interval}",
                         para.label_type, para.para_type)


# ------------------------------------------------------------------
# Loaders (identical to crossing_specificity.py)
# ------------------------------------------------------------------

def load_crossings(para, output_dir=None):
    """Load every *_crossings.csv saved by LabelRatioCurveAnalyzer for this
    group. Columns: curve, series, crossing_order, x, y."""
    d = crossings_dir(para, output_dir)
    paths = sorted(glob.glob(os.path.join(d, "*_crossings.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No *_crossings.csv found under {d}. "
            f"Run `python preparation.py --mode plot` for this para first."
        )
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["source_file"] = os.path.basename(p)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["symbol"] = para.symbol
    out["interval"] = para.interval
    out["label_type"] = para.label_type
    out["para_type"] = para.para_type
    return out


def load_self_eval(para, save_dir=None):
    d = train_dir(para, save_dir)
    return pd.read_csv(os.path.join(d, "self_eval_summary_mean_std.csv"))


def load_financial(para, save_dir=None):
    d = train_dir(para, save_dir)
    return pd.read_csv(os.path.join(d, "financial_return_summary_mean_std.csv"))


def _true_x(values, para_type):
    """Same rescale as crossing_specificity.py: horizon thresholds are
    stored as TRUE_HORIZON * 10 in self_eval/financial CSVs; crossing x
    values are already on the true horizon scale."""
    arr = np.asarray(values, dtype=float)
    if para_type == "horizon":
        return arr / 10.0
    return arr


METRIC_MAP = {
    "self_eval": {
        "loader": load_self_eval,
        "x_col": "threshold",
        "metric_cols": [
            "macro_f1_mean", "mcc_mean", "accuracy_mean",
            "p_pos_mean", "r_pos_mean", "f_pos_mean",
            "p_neg_mean", "r_neg_mean", "f_neg_mean",
        ],
    },
    "financial": {
        "loader": load_financial,
        "x_col": "threshold",
        "metric_cols": ["signal_avg_return_mean", "strategy_total_return_mean"],
    },
}


# ------------------------------------------------------------------
# Core method: smooth -> find turning points
# ------------------------------------------------------------------

def _auto_window_length(n, polyorder=3, target_frac=0.12, min_window=None):
    """
    Pick an odd Savitzky-Golay window length scaled to curve length, so
    short curves (e.g. ~18 points for XAUUSD-daily horizon sweeps) don't
    get an oversized window relative to their length, and long curves
    (e.g. 100 points for volatility sweeps) get enough smoothing to average
    out point-to-point sampling noise. `min_window` is enforced as a floor
    unless the curve itself is shorter than that.
    """
    if min_window is None:
        min_window = polyorder + 2
    w = max(min_window, int(round(n * target_frac)))
    if w % 2 == 0:
        w += 1
    w = min(w, n - (1 - n % 2))  # savgol requires window_length <= n (odd)
    w = max(w, polyorder + 2 if (polyorder + 2) % 2 else polyorder + 3)
    return w


def find_turning_points(x, y, window_length=None, polyorder=3,
                         prominence=None, edge_margin_frac=0.05):
    """
    Smooth (x, y) with a Savitzky-Golay filter and return the interior
    local minima/maxima of the smoothed curve as (x_location, kind) pairs,
    excluding extrema within `edge_margin_frac` of either boundary (these
    are usually filter edge artifacts rather than genuine turns).

    prominence: minimum prominence (in y-units) for a local extremum to
    count; if None, defaults to a small fraction of the curve's own y-range
    so flat/near-constant curves don't report spurious noise-level bumps.

    Returns (turning_points, y_smooth, window_length_used) where
    turning_points is a list of (x_value, 'min'|'max') tuples, sorted by x.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)

    if window_length is None:
        window_length = _auto_window_length(n, polyorder=polyorder)
    if prominence is None:
        y_range = np.nanmax(y) - np.nanmin(y)
        prominence = max(y_range * 0.03, 1e-6)

    y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)

    minima_idx, _ = find_peaks(-y_smooth, prominence=prominence)
    maxima_idx, _ = find_peaks(y_smooth, prominence=prominence)

    all_idx = np.concatenate([minima_idx, maxima_idx]) if (len(minima_idx) + len(maxima_idx)) > 0 else np.array([], dtype=int)
    kinds = ["min"] * len(minima_idx) + ["max"] * len(maxima_idx)
    ord2 = np.argsort(all_idx)
    all_idx, kinds = all_idx[ord2], [kinds[i] for i in ord2]

    margin = edge_margin_frac * (x.max() - x.min())
    turning_points = [
        (float(x[i]), k) for i, k in zip(all_idx, kinds)
        if (x[i] - x.min()) > margin and (x.max() - x[i]) > margin
    ]
    return turning_points, y_smooth, window_length


# ------------------------------------------------------------------
# Formatting helpers for the side-by-side table
# ------------------------------------------------------------------

_CURVE_ABBR = {"ratio": "ratio", "first_derivative": "d1", "second_derivative": "d2"}
_SERIES_ABBR = {"positive_vs_gaussian": "pos", "negative_vs_gaussian": "neg",
                "empirical_vs_gaussian": "emp"}


def _format_turning_points(turning_points):
    """'3.000(min); 7.900(max)' -- empty string if none found."""
    return "; ".join(f"{tx:.3f}({k})" for tx, k in turning_points)


def _format_crossings(crossings_df):
    """
    'x0(curve_abbr,series_abbr,order); x1(...); ...' sorted by x, so the
    reader can see both WHERE each crossing is and WHAT kind of crossing it
    is (raw ratio / 1st derivative / 2nd derivative; empirical / positive /
    negative vs the Gaussian reference; 1st or 2nd crossing along the
    parameter sweep) without needing a separate lookup table.
    """
    if crossings_df.empty:
        return ""
    rows = crossings_df.sort_values("x")
    parts = []
    for r in rows.itertuples():
        curve_abbr = _CURVE_ABBR.get(r.curve, r.curve)
        series_abbr = _SERIES_ABBR.get(getattr(r, "series", "empirical_vs_gaussian"),
                                        getattr(r, "series", ""))
        order = getattr(r, "crossing_order", "")
        parts.append(f"{r.x:.3f}({curve_abbr},{series_abbr},{order})")
    return "; ".join(parts)


# ------------------------------------------------------------------
# Full pipeline for one (table, metric, model) combination
# ------------------------------------------------------------------

def analyze_group(para, table_name, metric_col, group_filter,
                   output_dir=None, save_dir=None,
                   window_length=None, polyorder=3, prominence=None,
                   edge_margin_frac=0.05):
    """
    Returns a single-row DataFrame: the performance curve's own turning
    points and the full list of label-proportion Gaussian-crossing points
    for this experimental group, side by side, for manual comparison.
    """
    cfg = METRIC_MAP[table_name]
    crossings = load_crossings(para, output_dir)
    summary = cfg["loader"](para, save_dir)

    for k, v in group_filter.items():
        summary = summary[summary[k] == v]
    if summary.empty:
        raise ValueError(f"No rows in {table_name} summary for filter {group_filter}")

    x_col = cfg["x_col"]
    summary = summary.sort_values(x_col)
    x = _true_x(summary[x_col].to_numpy(dtype=float), para.para_type)
    y = summary[metric_col].to_numpy(dtype=float)

    turning_points, _, window_used = find_turning_points(
        x, y, window_length=window_length, polyorder=polyorder,
        prominence=prominence, edge_margin_frac=edge_margin_frac,
    )

    row = {
        "n_points": len(x),
        "window_length_used": window_used,
        "turning_point_count": len(turning_points),
        "turning_points": _format_turning_points(turning_points),
        "crossing_point_count": len(crossings),
        "crossing_points": _format_crossings(crossings),
        "table": table_name,
        "metric": metric_col,
        "symbol": para.symbol,
        "interval": para.interval,
        "label_type": para.label_type,
        "para_type": para.para_type,
    }
    row.update(group_filter)
    return pd.DataFrame([row])


# ------------------------------------------------------------------
# Full pipeline over all models/metrics for one group, and over all groups
# ------------------------------------------------------------------

def run_full_analysis(para, models=("LogisticRegression", "DecisionTree", "LSTM"),
                       eval_mode="balanced", fee_rate=0.0, output_dir=None,
                       save_dir=None, save=True, verbose=True,
                       window_length=None, polyorder=3, prominence=None,
                       edge_margin_frac=0.05):
    all_results = []
    for model in models:
        for metric_col in METRIC_MAP["self_eval"]["metric_cols"]:
            try:
                r = analyze_group(
                    para, "self_eval", metric_col,
                    group_filter={"model": model, "eval_mode": eval_mode},
                    output_dir=output_dir, save_dir=save_dir,
                    window_length=window_length, polyorder=polyorder,
                    prominence=prominence, edge_margin_frac=edge_margin_frac,
                )
                all_results.append(r)
            except Exception as e:
                if verbose:
                    print(f"[skip] self_eval {model}/{metric_col}: {e}")

        for metric_col in METRIC_MAP["financial"]["metric_cols"]:
            try:
                r = analyze_group(
                    para, "financial", metric_col,
                    group_filter={"model": model, "fee_rate": fee_rate},
                    output_dir=output_dir, save_dir=save_dir,
                    window_length=window_length, polyorder=polyorder,
                    prominence=prominence, edge_margin_frac=edge_margin_frac,
                )
                all_results.append(r)
            except Exception as e:
                if verbose:
                    print(f"[skip] financial {model}/{metric_col}: {e}")

    if not all_results:
        raise RuntimeError(
            f"No results produced for "
            f"{para.symbol}_{para.interval}/{para.label_type}/{para.para_type}"
        )

    master = pd.concat(all_results, ignore_index=True)

    if save:
        out_dir = train_dir(para, save_dir)
        master_path = os.path.join(out_dir, "turning_point_vs_crossing_table.csv")
        master.to_csv(master_path, index=False)
        print(f"table saved: {master_path}")

    return master


EXPERIMENTAL_GROUPS = [
    common.BTC_15m_fthl_volatility, common.BTC_15m_fthl_horizon,
    common.BTC_15m_tbm_volatility, common.BTC_15m_tbm_horizon,
    common.XAUUSD_15m_fthl_volatility, common.XAUUSD_15m_fthl_horizon,
    common.XAUUSD_15m_tbm_volatility, common.XAUUSD_15m_tbm_horizon,
    common.XAUUSD_1d_fthl_volatility, common.XAUUSD_1d_fthl_horizon,
    common.XAUUSD_1d_tbm_volatility, common.XAUUSD_1d_tbm_horizon,
] if common is not None else []

REPORT_DIR = (
    os.path.join(common.OUTPUT_DIR, "turning_point_specificity_report")
    if common is not None else None
)


def run_all_groups(paras=None, models=("LogisticRegression", "DecisionTree", "LSTM"),
                    eval_mode="balanced", fee_rate=0.0, output_dir=None,
                    save_dir=None, report_dir=None,
                    window_length=None, polyorder=3, prominence=None,
                    edge_margin_frac=0.05):
    paras = paras or EXPERIMENTAL_GROUPS
    report_dir = report_dir or REPORT_DIR
    os.makedirs(report_dir, exist_ok=True)

    all_masters = []
    for para in paras:
        try:
            master = run_full_analysis(
                para, models=models, eval_mode=eval_mode, fee_rate=fee_rate,
                output_dir=output_dir, save_dir=save_dir, save=False, verbose=False,
                window_length=window_length, polyorder=polyorder,
                prominence=prominence, edge_margin_frac=edge_margin_frac,
            )
            all_masters.append(master)
        except Exception as e:
            print(f"[skip group] {para.symbol}_{para.interval}/"
                  f"{para.label_type}/{para.para_type}: {e}")

    if not all_masters:
        raise RuntimeError("No experimental group produced results.")

    master_all = pd.concat(all_masters, ignore_index=True)

    master_path = os.path.join(report_dir, "turning_point_vs_crossing_table_all_groups.csv")
    master_all.to_csv(master_path, index=False)
    print(f"table (all groups) saved: {master_path}")

    return master_all


if __name__ == "__main__":
    paras = [
        common.BTC_15m_fthl_volatility,
        common.BTC_15m_fthl_horizon,
        common.BTC_15m_tbm_volatility,
        common.BTC_15m_tbm_horizon,

        common.XAUUSD_15m_fthl_volatility,
        common.XAUUSD_15m_fthl_horizon,
        common.XAUUSD_15m_tbm_volatility,
        common.XAUUSD_15m_tbm_horizon,

        common.XAUUSD_1d_fthl_volatility,
        common.XAUUSD_1d_fthl_horizon,
        common.XAUUSD_1d_tbm_volatility,
        common.XAUUSD_1d_tbm_horizon,
    ]
    master_all = run_all_groups(paras)
    print(master_all[["symbol", "interval", "label_type", "para_type", "model",
                       "table", "metric", "turning_points", "crossing_points"]])