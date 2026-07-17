"""
crossing_specificity.py
------------------------
Quantitatively test whether the (empirical label-ratio) x (Gaussian null)
crossing points found by `plot_label_ratio_sweep.LabelRatioCurveAnalyzer`
coincide with structural changes in the downstream ML performance curves
(self-eval macro-F1 / MCC / accuracy / per-class precision-recall-F1,
financial signal_avg_return / strategy_total_return, ...).

It does NOT recompute crossings — it reads the CSVs that
`LabelRatioCurveAnalyzer.plot_all_label_ratio_curves()` already saved
(`*_crossings.csv` under regime_discovery_output/...), and joins them
against the CSVs that `train.py` already saved under
`common.TRAIN_OUT_DIR/{symbol}_{interval}/{label_type}/{para_type}/`.

Scope of this module: ONLY the crossing-point specificity question ("does
this statistically-defined point in the label-ratio structure coincide with
a structural change in model performance?"). Whether different models agree
with each other is a separate question handled in compose_report_images.py
(model_curve_similarity section), which compares the raw performance curves
directly rather than these crossing-point test statistics.

Four independent, complementary quantitative checks are provided so that
"the crossing is special" is not asserted from a single method:

1. piecewise_breakpoint       - a *data-driven* single-breakpoint 2-segment
                                 linear fit on the performance curve, found
                                 with NO knowledge of the crossing x. If this
                                 independently detected breakpoint lands near
                                 the crossing x, that's real evidence.
2. bootstrap_breakpoint_ci    - resamples the 20 independent training runs
                                 (run_id) to get a confidence interval for
                                 that data-driven breakpoint, so we can test
                                 whether the crossing x falls inside the CI
                                 rather than eyeballing a single mean curve.
3. chow_test / chow_permutation_test
                               - classic structural-break F-test: does a
                                 piecewise-linear fit with a break AT the
                                 known crossing x fit significantly better
                                 than a single line through the whole curve?
                                 The raw chow_p fires almost everywhere on a
                                 curved/monotonic curve, so chow_permutation_test
                                 ranks x0's F-statistic against random split
                                 points to get a baseline-corrected p-value
                                 (chow_rank_p) — use that one, not chow_p.
4. permutation_extremum_test  - is the nearest local extremum (peak/trough)
                                 of the performance curve closer to the
                                 crossing x than to a uniformly random x in
                                 the same range? (permutation p-value)

`run_full_analysis()` loops this across every model x metric combination for
ONE experimental group. `run_all_groups()` loops that across all 12
dissertation experimental groups and pools the result, for the cross-group
consistency check discussed in "Comparative Analysis of Label Effects".
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
sys.path.append(current_work_dir)
from data_process import common


# ------------------------------------------------------------------
# Path helpers (mirror the layout produced by preparation.py / train.py)
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
# Loaders
# ------------------------------------------------------------------

def load_crossings(para, output_dir=None):
    """Load every *_crossings.csv saved by LabelRatioCurveAnalyzer for this
    (symbol, interval, label_type, para_type) combination."""
    d = crossings_dir(para, output_dir)
    paths = sorted(glob.glob(os.path.join(d, "*_crossings.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No *_crossings.csv found under {d}. "
            f"Run `python preparation.py --mode plot` for this para first "
            f"(see plot_label_distribution in preparation.py)."
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
    summary = pd.read_csv(os.path.join(d, "self_eval_summary_mean_std.csv"))
    raw_path = os.path.join(d, "self_eval_all_runs.csv")
    raw = pd.read_csv(raw_path) if os.path.exists(raw_path) else None
    return summary, raw


def load_financial(para, save_dir=None):
    d = train_dir(para, save_dir)
    summary = pd.read_csv(os.path.join(d, "financial_return_summary_mean_std.csv"))
    raw_path = os.path.join(d, "financial_return_all_runs.csv")
    raw = pd.read_csv(raw_path) if os.path.exists(raw_path) else None
    return summary, raw


def _true_x(values, para_type):
    """
    Mirror plot_results.py's `_scale_x()`.

    preparation.py's batch_label mode names horizon label columns with
    `f"h{int(round(t_range * 10)):02d}"` (the same x10 convention used for
    volatility multipliers), so the `threshold` column persisted by train.py
    into self_eval / financial CSVs stores TRUE_HORIZON * 10 whenever
    para_type == "horizon" (e.g. horizon=16 is stored as 160).

    The crossing x values produced by LabelRatioCurveAnalyzer, however, are
    on the TRUE horizon scale (1-80), because preparation.py's
    `plot_label_distribution()` sweeps `parameter_range = np.arange(1, 81, 1)`
    directly as `predict_num`.

    Without this rescale, every self_eval/financial threshold is compared
    against crossing x on the wrong scale (off by ~10x), which silently
    produces meaningless Chow/permutation/breakpoint results (large,
    suspiciously uniform gaps; 0% significance everywhere) instead of an
    error. Volatility thresholds need no rescale.
    """
    arr = np.asarray(values, dtype=float)
    if para_type == "horizon":
        return arr / 10.0
    return arr


# threshold column + metric columns available in each summary table
# (names copied directly from train.py's groupby(...).agg(...) calls)
METRIC_MAP = {
    "self_eval": {
        "loader": load_self_eval,
        "x_col": "threshold",
        "group_cols": ["model", "eval_mode"],
        "metric_cols": [
            "macro_f1_mean", "mcc_mean", "accuracy_mean",
            "p_pos_mean", "r_pos_mean", "f_pos_mean",
            "p_neg_mean", "r_neg_mean", "f_neg_mean",
        ],
    },
    "financial": {
        "loader": load_financial,
        "x_col": "threshold",
        "group_cols": ["model", "fee_rate"],
        "metric_cols": ["signal_avg_return_mean", "strategy_total_return_mean"],
    },
}


# ------------------------------------------------------------------
# 1. Data-driven breakpoint (no knowledge of the crossing x)
# ------------------------------------------------------------------

def piecewise_breakpoint(x, y, min_seg=4):
    """Grid-search the single split point that minimizes the summed SSE of
    two independent least-squares lines fit on either side. Returns the
    x-value of the best split plus diagnostic slopes."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)
    if n < 2 * min_seg + 1:
        return {"x_break": np.nan, "sse": np.nan,
                "slope_left": np.nan, "slope_right": np.nan}

    best = None
    for i in range(min_seg, n - min_seg):
        x1, y1 = x[:i + 1], y[:i + 1]
        x2, y2 = x[i:], y[i:]
        b1 = np.polyfit(x1, y1, 1)
        b2 = np.polyfit(x2, y2, 1)
        sse = (np.sum((np.polyval(b1, x1) - y1) ** 2) +
               np.sum((np.polyval(b2, x2) - y2) ** 2))
        if best is None or sse < best["sse"]:
            best = {"index": i, "x_break": float(x[i]), "sse": float(sse),
                     "slope_left": float(b1[0]), "slope_right": float(b2[0])}
    return best


def bootstrap_breakpoint_ci(raw_df, x_col, y_col, group_filter, para_type,
                             n_boot=300, min_seg=4, random_state=0):
    """Resample run_id (the 20 independent repeats) to build a 95% CI for
    the data-driven breakpoint above, so we can test whether the Gaussian
    crossing x lands inside it instead of trusting a single mean curve."""
    rng = np.random.default_rng(random_state)
    sub = raw_df.copy()
    for k, v in group_filter.items():
        sub = sub[sub[k] == v]
    if sub.empty or "run_id" not in sub.columns:
        return None

    run_ids = sub["run_id"].unique()
    breaks = []
    for _ in range(n_boot):
        sampled = rng.choice(run_ids, size=len(run_ids), replace=True)
        boot_df = sub[sub["run_id"].isin(sampled)]
        agg = boot_df.groupby(x_col)[y_col].mean().reset_index()
        if len(agg) < 2 * min_seg + 1:
            continue
        x_true = _true_x(agg[x_col].values, para_type)
        bp = piecewise_breakpoint(x_true, agg[y_col].values, min_seg=min_seg)
        if np.isfinite(bp["x_break"]):
            breaks.append(bp["x_break"])

    if len(breaks) < 20:
        return None
    breaks = np.array(breaks)
    return {
        "median": float(np.median(breaks)),
        "ci_low": float(np.percentile(breaks, 2.5)),
        "ci_high": float(np.percentile(breaks, 97.5)),
        "n_boot_used": len(breaks),
    }


# ------------------------------------------------------------------
# 2. Chow test at the KNOWN crossing x, with a permutation baseline
# ------------------------------------------------------------------

def chow_test(x, y, x0):
    """Structural-break F-test: does splitting the linear fit exactly at x0
    reduce SSE significantly more than sampling noise would predict?"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]

    def fit_sse(xx, yy):
        if len(xx) < 3:
            return None, len(xx)
        b = np.polyfit(xx, yy, 1)
        resid = yy - np.polyval(b, xx)
        return float(np.sum(resid ** 2)), len(xx)

    sse_pool, n = fit_sse(x, y)
    left, right = x <= x0, x > x0
    sse1, n1 = fit_sse(x[left], y[left])
    sse2, n2 = fit_sse(x[right], y[right])

    if None in (sse_pool, sse1, sse2) or n1 < 3 or n2 < 3:
        return {"f_stat": np.nan, "p_value": np.nan, "n1": n1, "n2": n2}

    k = 2  # slope + intercept per segment
    dof1, dof2 = k, n1 + n2 - 2 * k
    if dof2 <= 0:
        return {"f_stat": np.nan, "p_value": np.nan, "n1": n1, "n2": n2}

    f_stat = ((sse_pool - (sse1 + sse2)) / dof1) / ((sse1 + sse2) / dof2)
    p_value = 1 - stats.f.cdf(f_stat, dof1, dof2) if np.isfinite(f_stat) else np.nan
    return {"f_stat": float(f_stat), "p_value": float(p_value), "n1": int(n1), "n2": int(n2)}


def chow_permutation_test(x, y, x0, n_perm=500, random_state=0):
    """
    Raw chow_test() p-values are misleading on their own: on a curve that is
    simply monotonic/curved (true of almost every metric-vs-parameter curve
    in this study), splitting at ANY interior point tends to fit better than
    one global line, so chow_p < 0.05 fires almost everywhere and says
    nothing about x0 specifically.

    This asks the sharper question: is x0 among the BEST split points on
    this curve, or just a typical one? It compares the Chow F-statistic at
    x0 against the F-statistics obtained at `n_perm` uniformly-sampled
    reference split points in the same range, and returns the fraction of
    reference points that are AT LEAST as good a split as x0 (i.e. a
    rank-based p-value). A LOW value means x0 stands out as an unusually
    good split point relative to the rest of the curve; a value near 1
    means x0 is an unremarkable split point on an already-curved line.
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    observed = chow_test(x, y, x0)
    if not np.isfinite(observed["f_stat"]):
        return {"rank_p": np.nan, "observed_f": np.nan, "n_perm_used": 0}

    null_x0 = rng.uniform(np.min(x), np.max(x), size=n_perm)
    null_f = [r["f_stat"] for xx in null_x0
              if np.isfinite((r := chow_test(x, y, xx))["f_stat"])]
    null_f = np.array(null_f)
    if len(null_f) == 0:
        return {"rank_p": np.nan, "observed_f": observed["f_stat"], "n_perm_used": 0}

    rank_p = float(np.mean(null_f >= observed["f_stat"]))
    return {"rank_p": rank_p, "observed_f": observed["f_stat"], "n_perm_used": len(null_f)}


# ------------------------------------------------------------------
# 3. Nearest-extremum permutation test
# ------------------------------------------------------------------

def nearest_extremum_distance(x, y, x0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    dy = np.gradient(y, x)
    sign_change = np.where(np.diff(np.sign(dy)) != 0)[0]
    if len(sign_change) == 0:
        return np.nan
    extrema_x = x[sign_change]
    return float(np.min(np.abs(extrema_x - x0)))


def permutation_extremum_test(x, y, x0, n_perm=2000, random_state=0):
    """Is the nearest local peak/trough of the performance curve closer to
    the crossing x than it would be to a uniformly-random x in the same
    parameter range?"""
    rng = np.random.default_rng(random_state)
    observed = nearest_extremum_distance(x, y, x0)
    if np.isnan(observed):
        return {"observed_distance": np.nan, "p_value": np.nan, "n_perm": n_perm}
    null_x0 = rng.uniform(np.min(x), np.max(x), size=n_perm)
    null_d = np.array([nearest_extremum_distance(x, y, xx) for xx in null_x0])
    p = float(np.mean(null_d <= observed))
    return {"observed_distance": observed, "p_value": p, "n_perm": n_perm}


# ------------------------------------------------------------------
# 4. Full pipeline for one (table, metric, group) combination
# ------------------------------------------------------------------

def analyze_group(para, table_name, metric_col, group_filter,
                   output_dir=None, save_dir=None, n_boot=300):
    """
    table_name   : "self_eval" or "financial"
    metric_col   : e.g. "macro_f1_mean", "signal_avg_return_mean"
    group_filter : dict restricting rows, e.g. {"model": "LogisticRegression",
                   "eval_mode": "balanced"} or {"model": "LSTM", "fee_rate": 0.0}
    """
    cfg = METRIC_MAP[table_name]
    crossings = load_crossings(para, output_dir)
    summary, raw = cfg["loader"](para, save_dir)

    for k, v in group_filter.items():
        summary = summary[summary[k] == v]
    if summary.empty:
        raise ValueError(f"No rows in {table_name} summary for filter {group_filter}")

    x_col = cfg["x_col"]
    summary = summary.sort_values(x_col)
    x = _true_x(summary[x_col].to_numpy(dtype=float), para.para_type)
    y = summary[metric_col].to_numpy(dtype=float)

    data_bp = piecewise_breakpoint(x, y)
    raw_y_col = metric_col.replace("_mean", "")
    boot = (bootstrap_breakpoint_ci(raw, x_col, raw_y_col, group_filter,
                                     para.para_type, n_boot=n_boot)
            if raw is not None else None)

    x_range = float(np.max(x) - np.min(x)) if len(x) > 1 else np.nan

    rows = []
    for _, r in crossings.iterrows():
        x0 = float(r["x"])
        chow = chow_test(x, y, x0)
        chow_perm = chow_permutation_test(x, y, x0)
        perm = permutation_extremum_test(x, y, x0)

        # A wide bootstrap CI covers more of the parameter range, so ANY x0
        # -- crossing point or not -- lands inside it more often just from
        # width. ci_chance_rate is what a uniformly random x0 would hit
        # purely from that width; boot_hit_excess is the observed hit (0/1)
        # minus that chance rate.
        if boot and np.isfinite(x_range) and x_range > 0:
            ci_width = boot["ci_high"] - boot["ci_low"]
            ci_chance_rate = float(np.clip(ci_width / x_range, 0.0, 1.0))
            hit = 1.0 if (boot["ci_low"] <= x0 <= boot["ci_high"]) else 0.0
            boot_hit_excess = hit - ci_chance_rate
        else:
            ci_chance_rate = np.nan
            boot_hit_excess = np.nan

        rows.append({
            "crossing_curve": r["curve"],
            "crossing_series": r.get("series", "empirical_vs_gaussian"),
            "crossing_order": r.get("crossing_order"),
            "x0_crossing": x0,
            "data_driven_breakpoint": data_bp["x_break"],
            "breakpoint_gap": (abs(data_bp["x_break"] - x0)
                                if np.isfinite(data_bp["x_break"]) else np.nan),
            "chow_f": chow["f_stat"],
            "chow_p": chow["p_value"],
            "chow_rank_p": chow_perm["rank_p"],
            "extremum_perm_p": perm["p_value"],
            "extremum_distance": perm["observed_distance"],
            "boot_ci_low": boot["ci_low"] if boot else np.nan,
            "boot_ci_high": boot["ci_high"] if boot else np.nan,
            "x0_in_boot_ci": (boot["ci_low"] <= x0 <= boot["ci_high"]) if boot else np.nan,
            "ci_chance_rate": ci_chance_rate,
            "boot_hit_excess": boot_hit_excess,
        })

    out = pd.DataFrame(rows)
    out["table"] = table_name
    out["metric"] = metric_col
    for k, v in group_filter.items():
        out[k] = v
    out["symbol"] = para.symbol
    out["interval"] = para.interval
    out["label_type"] = para.label_type
    out["para_type"] = para.para_type

    return out


# ------------------------------------------------------------------
# 5. Sanity check + full pipeline over all models/metrics for one group
# ------------------------------------------------------------------

def sanity_check_scales(para, output_dir=None, save_dir=None):
    """Print the x-range of crossings vs self_eval vs financial tables so a
    unit mismatch (like the horizon x10 issue) is caught immediately instead
    of silently producing a null result."""
    crossings = load_crossings(para, output_dir)
    self_summary, _ = load_self_eval(para, save_dir)
    fin_summary, _ = load_financial(para, save_dir)

    print(f"[sanity check] crossing x range      : "
          f"{crossings['x'].min():.3f} .. {crossings['x'].max():.3f}")
    print(f"[sanity check] self_eval threshold   : "
          f"{self_summary['threshold'].min():.3f} .. {self_summary['threshold'].max():.3f} "
          f"(rescaled -> {_true_x(self_summary['threshold'], para.para_type).min():.3f} .. "
          f"{_true_x(self_summary['threshold'], para.para_type).max():.3f})")
    print(f"[sanity check] financial threshold   : "
          f"{fin_summary['threshold'].min():.3f} .. {fin_summary['threshold'].max():.3f} "
          f"(rescaled -> {_true_x(fin_summary['threshold'], para.para_type).min():.3f} .. "
          f"{_true_x(fin_summary['threshold'], para.para_type).max():.3f})")
    print("If the rescaled ranges above don't roughly match the crossing x "
          "range, there is still a unit mismatch to track down before "
          "trusting chow_p / extremum_perm_p / breakpoint_gap.")


def _consistency_summary(master):
    """For each crossing type (curve/series/order), what fraction of
    (group, model, metric) tests show a significant structural break or a
    proximate local extremum? Works identically whether `master` covers one
    experimental group or all of them concatenated together.

    frac_chow_significant is the RAW Chow test rate (chow_p < 0.05) and is
    included for transparency only — on a curved/monotonic performance
    curve this fires at almost every candidate split point, so a high value
    here does NOT by itself indicate that x0 is special. Use
    frac_chow_rank_significant instead: it requires x0's Chow F-statistic to
    beat at least 95% of random split points in the same range, which is
    the baseline-corrected version of the same test.
    """
    return (
            master.groupby(["crossing_curve", "crossing_series", "crossing_order"])
            .agg(
                n_tests=("chow_p", "size"),
                frac_chow_significant=("chow_p", lambda s: float(np.mean(s < 0.05))),
                frac_chow_rank_significant=("chow_rank_p", lambda s: float(np.mean(s < 0.05)) if s.notna().any() else np.nan),
                frac_extremum_significant=("extremum_perm_p", lambda s: float(np.mean(s < 0.05))),
                median_breakpoint_gap=("breakpoint_gap", "median"),
                frac_x0_in_boot_ci=("x0_in_boot_ci", lambda s: float(np.mean(s.dropna())) if s.notna().any() else np.nan),
                mean_ci_chance_rate=("ci_chance_rate", lambda s: float(np.mean(s.dropna())) if s.notna().any() else np.nan),
                mean_boot_hit_excess=("boot_hit_excess", lambda s: float(np.mean(s.dropna())) if s.notna().any() else np.nan),
            )
            .reset_index()
        )


def run_full_analysis(para, models=("LogisticRegression", "DecisionTree", "LSTM"),
                       eval_mode="balanced", fee_rate=0.0, output_dir=None,
                       save_dir=None, n_boot=300, save=True, verbose=True):
    """Run every self_eval metric x model and every financial metric x model
    for ONE experimental group (one symbol/interval/label_type/para_type),
    concatenate into a single tidy DataFrame.

    If save=True (default), writes exactly 2 files into this group's
    train_dir: crossing_specificity_master.csv and
    crossing_specificity_consistency_summary.csv. Set save=False when
    calling this from run_all_groups(), which does its own single
    consolidated save across all groups instead.

    Returns (master, consistency).
    """
    if verbose:
        sanity_check_scales(para, output_dir, save_dir)
    all_results = []

    for model in models:
        for metric_col in METRIC_MAP["self_eval"]["metric_cols"]:
            try:
                r = analyze_group(
                    para, "self_eval", metric_col,
                    group_filter={"model": model, "eval_mode": eval_mode},
                    output_dir=output_dir, save_dir=save_dir, n_boot=n_boot,
                )
                all_results.append(r)
            except Exception as e:
                print(f"[skip] self_eval {model}/{metric_col}: {e}")

        for metric_col in METRIC_MAP["financial"]["metric_cols"]:
            try:
                r = analyze_group(
                    para, "financial", metric_col,
                    group_filter={"model": model, "fee_rate": fee_rate},
                    output_dir=output_dir, save_dir=save_dir, n_boot=n_boot,
                )
                all_results.append(r)
            except Exception as e:
                print(f"[skip] financial {model}/{metric_col}: {e}")

    if not all_results:
        raise RuntimeError(
            f"No results produced for "
            f"{para.symbol}_{para.interval}/{para.label_type}/{para.para_type} "
            f"— check that crossings/train CSVs exist."
        )

    master = pd.concat(all_results, ignore_index=True)
    consistency = _consistency_summary(master)

    if save:
        out_dir = train_dir(para, save_dir)
        master_path = os.path.join(out_dir, "crossing_specificity_master.csv")
        consistency_path = os.path.join(out_dir, "crossing_specificity_consistency_summary.csv")
        master.to_csv(master_path, index=False)
        consistency.to_csv(consistency_path, index=False)
        print(f"master results saved: {master_path}")
        print(f"consistency summary saved: {consistency_path}")

    return master, consistency


# ------------------------------------------------------------------
# 6. All 12 experimental groups -> exactly 2 global CSVs
# ------------------------------------------------------------------

EXPERIMENTAL_GROUPS = [
    common.BTC_15m_fthl_volatility, common.BTC_15m_fthl_horizon,
    common.BTC_15m_tbm_volatility, common.BTC_15m_tbm_horizon,
    common.XAUUSD_15m_fthl_volatility, common.XAUUSD_15m_fthl_horizon,
    common.XAUUSD_15m_tbm_volatility, common.XAUUSD_15m_tbm_horizon,
    common.XAUUSD_1d_fthl_volatility, common.XAUUSD_1d_fthl_horizon,
    common.XAUUSD_1d_tbm_volatility, common.XAUUSD_1d_tbm_horizon,
]

REPORT_DIR = os.path.join(common.OUTPUT_DIR, "crossing_specificity_report")


def run_all_groups(paras=None, models=("LogisticRegression", "DecisionTree", "LSTM"),
                    eval_mode="balanced", fee_rate=0.0, output_dir=None,
                    save_dir=None, n_boot=300, report_dir=None):
    """Run run_full_analysis() over every experimental group (default: all
    12 groups from the dissertation) WITHOUT letting each group save its own
    files, then write exactly 2 consolidated CSVs covering every group:

        crossing_specificity_master_all_groups.csv
        crossing_specificity_consistency_all_groups.csv

    The consistency table's rows are per (crossing_curve, crossing_series,
    crossing_order) POOLED across all groups/models/metrics — this is the
    cross-group consistency check discussed in the dissertation's
    Comparative Analysis section.
    """
    paras = paras or EXPERIMENTAL_GROUPS
    report_dir = report_dir or REPORT_DIR
    os.makedirs(report_dir, exist_ok=True)

    all_masters = []
    for para in paras:
        try:
            master, _ = run_full_analysis(
                para, models=models, eval_mode=eval_mode, fee_rate=fee_rate,
                output_dir=output_dir, save_dir=save_dir, n_boot=n_boot,
                save=False, verbose=False,
            )
            all_masters.append(master)
        except Exception as e:
            print(
                f"[skip group] {para.symbol}_{para.interval}/"
                f"{para.label_type}/{para.para_type}: {e}"
            )

    if not all_masters:
        raise RuntimeError("No experimental group produced results — check output/ paths.")

    master_all = pd.concat(all_masters, ignore_index=True)
    consistency_all = _consistency_summary(master_all)

    master_path = os.path.join(report_dir, "crossing_specificity_master_all_groups.csv")
    consistency_path = os.path.join(report_dir, "crossing_specificity_consistency_all_groups.csv")
    master_all.to_csv(master_path, index=False)
    consistency_all.to_csv(consistency_path, index=False)
    print(f"master results (all groups) saved: {master_path}")
    print(f"consistency summary (all groups) saved: {consistency_path}")

    return master_all, consistency_all


if __name__ == "__main__":
    # Single group (2 files, saved under that group's own train_dir):
    #   master, consistency = run_full_analysis(common.BTC_15m_fthl_horizon)

    # All 12 experimental groups pooled into exactly 2 files under
    # output/crossing_specificity_report/:
    master_all, consistency_all = run_all_groups()
    print(consistency_all)