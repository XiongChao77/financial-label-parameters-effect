#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compose a report image per BaseDefine config.

For each config in BASE_DEFINES, collects the images listed in IMAGE_SPECS
from output/regime_discovery_output/... and output/train/..., stitches them
into one big grid image per model, and appends two extra summary tables:

  1. crossing_specificity_summary.png  - does the label-ratio structure's
     Gaussian-crossing points coincide with structural changes in the
     performance curves? (delegates the actual test statistics to
     crossing_specificity.py)

  2. model_curve_similarity.png        - do LogisticRegression / DecisionTree
     / LSTM show the SAME curve for each metric in IMAGE_SPECS, so that one
     model could be reported as representative of the whole experiment?
     This compares the raw numeric curves directly (not the crossing-point
     test statistics), reading the same CSVs the PNGs above were plotted
     from.
"""
import itertools
import os
import sys

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
sys.path.append(current_work_dir)
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image, ImageDraw, ImageFont

from data_process import common
from analysis import crossing_specificity

# =====================================================
# Hardcoded images to collect for each config.
# Each entry: (title, root, relative_path_template)
#   root == "regime" -> output/regime_discovery_output/{symbol}_{interval}/{label_type}/{para_type}/<relative_path>
#   root == "train"  -> output/train/{symbol}_{interval}/{label_type}/{para_type}/<relative_path>
# relative_path_template may use {label_type_lower} / {para_type} / {model} placeholders.
# =====================================================

IMAGE_SPECS: List[Tuple[str, str, str]] = [
    # ("Label Distribution Overview", "regime", "distribution_overview.png"),
    ("Positive/Negative Ratio", "regime", "positive_negative_ratio.png"),
    ("Positive/Negative Ratio D1", "regime", "positive_negative_d1.png"),
    ("Positive/Negative Ratio D2", "regime", "positive_negative_d2.png"),
    ("Macro F1 (Balanced)", "train", "{model}/macro_f1_balanced.png"),
    ("Accuracy (Balanced)", "train", "{model}/accuracy_balanced.png"),
    ("MCC (Balanced)", "train", "{model}/mcc_balanced.png"),
    ("Positive F1 (Balanced)", "train", "{model}/positive_f1_balanced.png"),
    ("Positive Precision (Balanced)", "train", "{model}/positive_precision_balanced.png"),
    ("Positive Recall (Balanced)", "train", "{model}/positive_recall_balanced.png"),
    ("Negative F1 (Balanced)", "train", "{model}/negative_f1_balanced.png"),
    ("Negative Precision (Balanced)", "train", "{model}/negative_precision_balanced.png"),
    ("Negative Recall (Balanced)", "train", "{model}/negative_recall_balanced.png"),
    ("Macro F1 (Raw)", "train", "{model}/macro_f1_raw.png"),
    ("Accuracy (Raw)", "train", "{model}/accuracy_raw.png"),
    ("Positive F1 (Raw)", "train", "{model}/positive_f1_raw.png"),
    ("Positive Precision (Raw)", "train", "{model}/positive_precision_raw.png"),
    ("Positive Recall (Raw)", "train", "{model}/positive_recall_raw.png"),
    ("Negative F1 (Raw)", "train", "{model}/negative_f1_raw.png"),
    ("Negative Precision (Raw)", "train", "{model}/negative_precision_raw.png"),
    ("Negative Recall (Raw)", "train", "{model}/negative_recall_raw.png"),
    ("MCC (Raw)", "train", "{model}/mcc_raw.png"),
    ("Financial Signal Avg Return (fee=0)", "train", "{model}/financial_return/financial_signal_avg_return_fee_0.000000.png"),
    ("Financial Strategy Total Return (fee=0)", "train", "{model}/financial_return/financial_strategy_total_return_fee_0.000000.png"),
]
MODELS = ['DecisionTree', 'LogisticRegression', 'LSTM']

N_COLS = 3
THUMB_WIDTH = 900
REPORT_OUTPUT_DIR = os.path.join(common.OUTPUT_DIR, "composed_report")


def resolve_base_dir(pre_para: common.BaseDefine, root: str) -> str:
    symbol_interval = f"{pre_para.symbol}_{pre_para.interval}"
    if root == "regime":
        return os.path.join(common.OUTPUT_DIR, "regime_discovery_output", symbol_interval, pre_para.label_type, pre_para.para_type)
    elif root == "train":
        return os.path.join(common.OUTPUT_DIR, "train", symbol_interval, pre_para.label_type, pre_para.para_type)
    raise ValueError(f"Unknown image root: {root}")


def collect_images(pre_para: common.BaseDefine, model: str) -> List[Tuple[str, str]]:
    """Return (title, image_path) pairs for images that exist on disk."""
    found = []
    for title, root, rel_path_tpl in IMAGE_SPECS:
        rel_path = rel_path_tpl.format(
            label_type_lower=pre_para.label_type.lower(),
            para_type=pre_para.para_type,
            model=model,
        )
        img_path = os.path.join(resolve_base_dir(pre_para, root), rel_path)
        if os.path.isfile(img_path):
            found.append((title, img_path))
        else:
            print(f"[skip] image not found: {img_path}")
    return found


def compose_grid(
    images: List[Tuple[str, str]],
    n_cols: int,
    thumb_width: int,
    title_font_size: int = 22,
    figure_title: str = None,
    figure_title_font_size: int = 32,
) -> Image.Image:
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", title_font_size)
    except OSError:
        font = ImageFont.load_default()
    try:
        figure_title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", figure_title_font_size)
    except OSError:
        figure_title_font = ImageFont.load_default()

    title_h = title_font_size + 12
    figure_title_h = figure_title_font_size + 20 if figure_title else 0
    thumbs = []
    for title, path in images:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        thumb_h = int(h * thumb_width / w)
        thumbs.append((title, img.resize((thumb_width, thumb_h), Image.LANCZOS)))

    n_rows = (len(thumbs) + n_cols - 1) // n_cols
    cell_h = max(img.size[1] for _, img in thumbs) + title_h
    canvas_width = thumb_width * n_cols
    canvas = Image.new("RGB", (canvas_width, cell_h * n_rows + figure_title_h), "white")
    draw = ImageDraw.Draw(canvas)

    if figure_title:
        text_w = draw.textlength(figure_title, font=figure_title_font)
        draw.text(((canvas_width - text_w) / 2, 10), figure_title, fill="black", font=figure_title_font)

    for idx, (title, img) in enumerate(thumbs):
        row, col = divmod(idx, n_cols)
        x, y = col * thumb_width, row * cell_h + figure_title_h
        draw.text((x + 8, y + 4), title, fill="black", font=font)
        canvas.paste(img, (x, y + title_h))

    return canvas


# ======================================================================
# Generic PNG table renderer, shared by the crossing-specificity summary
# and the model-curve-similarity summary below.
# ======================================================================

def render_table_image(
    rows: List[List[str]],
    out_path: str,
    figure_title: str,
    col_width: int = 300,
    row_height: int = 40,
    font_size: int = 18,
    header_font_size: int = 20,
    figure_title_font_size: int = 30,
) -> str:
    """rows[0] is the header row; the rest are data rows, all already
    formatted as strings."""
    try:
        header_font = ImageFont.truetype("DejaVuSans-Bold.ttf", header_font_size)
        cell_font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        figure_title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", figure_title_font_size)
    except OSError:
        header_font = cell_font = figure_title_font = ImageFont.load_default()

    n_cols = len(rows[0])
    figure_title_h = figure_title_font_size + 20
    canvas_width = col_width * n_cols
    canvas_height = figure_title_h + row_height * len(rows)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    text_w = draw.textlength(figure_title, font=figure_title_font)
    draw.text(((canvas_width - text_w) / 2, 10), figure_title, fill="black", font=figure_title_font)

    for r, row_vals in enumerate(rows):
        y = figure_title_h + r * row_height
        font = header_font if r == 0 else cell_font
        bg = "#2b2b2b" if r == 0 else ("#f2f2f2" if r % 2 == 0 else "white")
        draw.rectangle([0, y, canvas_width, y + row_height], fill=bg)
        for c, text in enumerate(row_vals):
            x = c * col_width
            draw.text((x + 8, y + (row_height - font_size) / 2), text,
                       fill=("white" if r == 0 else "black"), font=font)
        draw.line([0, y + row_height, canvas_width, y + row_height], fill="#cccccc", width=1)

    canvas.save(out_path)
    return out_path


def _fmt(val) -> str:
    if isinstance(val, (bool, np.bool_)):
        return "YES" if val else "no"
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def render_consistency_table_image(consistency_df, out_path: str, figure_title: str) -> str:
    """Render crossing_specificity's per-crossing-point consistency summary
    as a plain PNG table."""
    display_cols = [
        "crossing_curve", "crossing_series", "crossing_order", "n_tests",
        "frac_chow_rank_significant", "frac_extremum_significant",
        "median_breakpoint_gap", "frac_x0_in_boot_ci",
    ]
    display_cols = [c for c in display_cols if c in consistency_df.columns]
    rows = [display_cols] + [
        [_fmt(row[c]) for c in display_cols]
        for _, row in consistency_df.iterrows()
    ]
    return render_table_image(rows, out_path, figure_title, col_width=320, row_height=44)


# ======================================================================
# Cross-model curve similarity: do the three models tell the same story?
# ======================================================================
# This is a DIFFERENT question from crossing_specificity.py: it does not
# care about crossing points or structural-break tests at all. It simply
# asks, for every metric that IMAGE_SPECS plots once per model, whether the
# three models' curves over the parameter sweep look alike. If they do
# (high correlation, low normalized RMSE) across most metrics, a single
# representative model can be reported instead of all three.

# ======================================================================
# Cross-model curve similarity: do the three models tell the same story?
# ======================================================================
# This is a DIFFERENT question from crossing_specificity.py: it does not
# care about crossing points or structural-break tests at all. It simply
# asks, for every curve that IMAGE_SPECS plots once per model, whether the
# three models' curves over the parameter sweep look alike. If they do
# (high correlation, low normalized RMSE) across most curves, a single
# representative model can be reported instead of all three.
#
# Each IMAGE_SPECS panel (e.g. "MCC (Raw)") overlays up to 3 curves --
# Self Eval, Cross Train, Cross Eval -- from 3 different summary tables with
# different x-axis column names and metric-column suffixes (see train.py):
#   self_eval_summary_mean_std.csv         x=threshold        {base}_mean
#   cross_train_batch_summary_mean_std.csv x=train_threshold  {base}_mean_mean
#   cross_eval_batch_summary_mean_std.csv  x=eval_threshold   {base}_mean_mean
# Financial panels only exist as a single curve (no cross_train/cross_eval
# analog), from financial_return_summary_mean_std.csv, x=threshold, {base}_mean.
# Comparing only the Self Eval curve (as an earlier version of this module
# did) silently ignores the Cross Train / Cross Eval curves in the same
# panel, which can visually dominate the image -- this section now scores
# every curve in every panel.

def _load_cross_train_summary(pre_para, save_dir=None):
    d = crossing_specificity.train_dir(pre_para, save_dir)
    return pd.read_csv(os.path.join(d, "cross_train_batch_summary_mean_std.csv"))


def _load_cross_eval_summary(pre_para, save_dir=None):
    d = crossing_specificity.train_dir(pre_para, save_dir)
    return pd.read_csv(os.path.join(d, "cross_eval_batch_summary_mean_std.csv"))


CURVE_SOURCES = {
    "self":        {"x_col": "threshold",       "suffix": "_mean",
                     "loader": lambda p, sd: crossing_specificity.load_self_eval(p, sd)[0]},
    "cross_train": {"x_col": "train_threshold",  "suffix": "_mean_mean", "loader": _load_cross_train_summary},
    "cross_eval":  {"x_col": "eval_threshold",   "suffix": "_mean_mean", "loader": _load_cross_eval_summary},
    "financial":   {"x_col": "threshold",        "suffix": "_mean",
                     "loader": lambda p, sd: crossing_specificity.load_financial(p, sd)[0]},
}

# title -> list of (curve_key, base_metric_name, extra_filter). base_metric
# names are shared across self/cross_train/cross_eval (train.py uses the
# same {base} for all three, just different suffixes).
_SELF_EVAL_FAMILY_METRICS = {
    "Macro F1":            "macro_f1",
    "Accuracy":            "accuracy",
    "MCC":                 "mcc",
    "Positive F1":         "f_pos",
    "Positive Precision":  "p_pos",
    "Positive Recall":     "r_pos",
    "Negative F1":         "f_neg",
    "Negative Precision":  "p_neg",
    "Negative Recall":     "r_neg",
}

IMAGE_CURVES = {}
for _name, _base in _SELF_EVAL_FAMILY_METRICS.items():
    for _mode in ("Balanced", "Raw"):
        _eval_mode = _mode.lower()
        _title = f"{_name} ({_mode})"
        _filter = {"eval_mode": _eval_mode}
        IMAGE_CURVES[_title] = [
            ("self", _base, _filter),
            ("cross_train", _base, _filter),
            ("cross_eval", _base, _filter),
        ]
IMAGE_CURVES["Financial Signal Avg Return (fee=0)"] = [
    ("financial", "signal_avg_return", {"fee_rate": 0.0}),
]
IMAGE_CURVES["Financial Strategy Total Return (fee=0)"] = [
    ("financial", "strategy_total_return", {"fee_rate": 0.0}),
]


def _per_model_image_specs():
    """Only the IMAGE_SPECS entries plotted once PER MODEL (root == 'train'
    with a {model} placeholder). The 3 'regime' label-ratio plots aren't
    model-specific and have nothing to compare across models."""
    return [(t, r, p) for t, r, p in IMAGE_SPECS if r == "train" and "{model}" in p]


def _load_curve(pre_para, curve_key, base_metric, model, extra_filter, save_dir=None):
    """The exact (x, y) series for one curve (self / cross_train / cross_eval
    / financial) within one IMAGE_SPECS panel, on the TRUE parameter scale
    (reuses crossing_specificity._true_x so every curve source stays in sync
    on the horizon x10 rescale)."""
    src = CURVE_SOURCES[curve_key]
    summary = src["loader"](pre_para, save_dir)
    metric_col = base_metric + src["suffix"]

    sub = summary[summary["model"] == model]
    for k, v in extra_filter.items():
        sub = sub[sub[k] == v]
    if sub.empty:
        raise ValueError(f"No rows for curve={curve_key}, model={model}, filter={extra_filter}")

    x_col = src["x_col"]
    sub = sub.sort_values(x_col)
    x = crossing_specificity._true_x(sub[x_col].to_numpy(dtype=float), pre_para.para_type)
    x = np.round(x, 6)  # avoid brittle float-equality merges downstream
    y = sub[metric_col].to_numpy(dtype=float)
    return pd.DataFrame({"x": x, "y": y})


def _curve_similarity(curve_a: pd.DataFrame, curve_b: pd.DataFrame) -> dict:
    """Align two (x, y) curves on x and compare shape (correlation) and
    absolute level (normalized RMSE)."""
    merged = pd.merge(curve_a, curve_b, on="x", suffixes=("_a", "_b")).dropna()
    n = len(merged)
    result = {"n_matched": n, "pearson_r": np.nan, "spearman_r": np.nan, "nrmse": np.nan}
    if n < 3:
        return result

    ya, yb = merged["y_a"].to_numpy(), merged["y_b"].to_numpy()
    if np.unique(ya).size > 1 and np.unique(yb).size > 1:
        result["pearson_r"] = float(stats.pearsonr(ya, yb)[0])
        result["spearman_r"] = float(stats.spearmanr(ya, yb)[0])

    rmse = float(np.sqrt(np.mean((ya - yb) ** 2)))
    span = float(max(ya.max(), yb.max()) - min(ya.min(), yb.min()))
    result["nrmse"] = rmse / span if span > 0 else np.nan
    return result


def analyze_curve_similarity(pre_para, models=tuple(MODELS), save_dir=None) -> pd.DataFrame:
    """For every curve (Self / Cross Train / Cross Eval / Financial) in every
    per-model panel of IMAGE_SPECS, compare every pair of models."""
    rows = []
    for title, root, rel_path_tpl in _per_model_image_specs():
        if title not in IMAGE_CURVES:
            print(f"[skip] no curve mapping for image title: {title!r} "
                  f"(add it to IMAGE_CURVES)")
            continue

        for curve_key, base_metric, extra_filter in IMAGE_CURVES[title]:
            curves = {}
            for model in models:
                try:
                    curves[model] = _load_curve(pre_para, curve_key, base_metric, model, extra_filter, save_dir)
                except Exception as e:
                    print(f"[skip] {title}/{curve_key}/{model}: {e}")

            for m1, m2 in itertools.combinations(models, 2):
                if m1 not in curves or m2 not in curves:
                    continue
                sim = _curve_similarity(curves[m1], curves[m2])
                rows.append({"image_title": title, "curve": curve_key,
                             "model_a": m1, "model_b": m2, **sim})

    out = pd.DataFrame(rows)
    out["symbol"] = pre_para.symbol
    out["interval"] = pre_para.interval
    out["label_type"] = pre_para.label_type
    out["para_type"] = pre_para.para_type
    return out


def overall_similarity_score(similarity_df, value_col="spearman_r", weight_col="n_matched") -> float:
    """Single number, weighted-mean spearman_r across every (image_title,
    curve, model pair) row, weighted by n_matched (how many x-points went
    into that row's correlation -- a correlation computed on more points is
    a more reliable estimate, so it gets more say in the average). This is
    the headline "overall model-to-model consistency" score."""
    df = similarity_df.dropna(subset=[value_col, weight_col])
    if df.empty or df[weight_col].sum() == 0:
        return float("nan")
    return float(np.average(df[value_col], weights=df[weight_col]))


def model_representativeness_score(similarity_df, models=tuple(MODELS),
                                    value_col="spearman_r", weight_col="n_matched") -> pd.DataFrame:
    """
    For each model, the weighted-mean spearman_r of every row where that
    model appears (as model_a or model_b) against the OTHER model in that
    row -- i.e. "on average, how closely does this model's curve track
    whichever other model it's compared against?" The model with the
    highest score is the most central / typical of the three, and is the
    natural pick when you can only report one model as representative of
    the whole experiment.

    This is a DIFFERENT question from "which model performs best" -- pair
    it with each model's own self_eval / financial numbers (see
    select_representative_model) before deciding.
    """
    rows = []
    for m in models:
        sub = similarity_df[(similarity_df["model_a"] == m) | (similarity_df["model_b"] == m)]
        sub = sub.dropna(subset=[value_col, weight_col])
        if sub.empty or sub[weight_col].sum() == 0:
            rows.append({"model": m, "n_rows": 0, "mean_similarity_to_others": np.nan})
            continue
        score = float(np.average(sub[value_col], weights=sub[weight_col]))
        rows.append({"model": m, "n_rows": int(len(sub)), "mean_similarity_to_others": score})
    return pd.DataFrame(rows).sort_values("mean_similarity_to_others", ascending=False).reset_index(drop=True)


def model_performance_score(pre_para, models=tuple(MODELS), eval_mode="balanced",
                             fee_rate=0.0, save_dir=None) -> pd.DataFrame:
    """For each model, its own mean performance averaged across the whole
    parameter sweep -- macro_f1 / mcc / accuracy from self_eval_summary
    (fixed eval_mode) and signal_avg_return / strategy_total_return from
    financial_return_summary (fixed fee_rate). This is the OTHER axis for
    picking a representative model: representativeness (above) says which
    model is most "typical"; this says which model actually performs best."""
    self_summary, _ = crossing_specificity.load_self_eval(pre_para, save_dir)
    fin_summary, _ = crossing_specificity.load_financial(pre_para, save_dir)

    rows = []
    for m in models:
        se = self_summary[(self_summary["model"] == m) & (self_summary["eval_mode"] == eval_mode)]
        fi = fin_summary[(fin_summary["model"] == m) & (fin_summary["fee_rate"] == fee_rate)]
        rows.append({
            "model": m,
            "mean_macro_f1": float(se["macro_f1_mean"].mean()) if not se.empty else np.nan,
            "mean_mcc": float(se["mcc_mean"].mean()) if not se.empty else np.nan,
            "mean_accuracy": float(se["accuracy_mean"].mean()) if not se.empty else np.nan,
            "mean_signal_avg_return": float(fi["signal_avg_return_mean"].mean()) if not fi.empty else np.nan,
            "mean_strategy_total_return": float(fi["strategy_total_return_mean"].mean()) if not fi.empty else np.nan,
        })
    return pd.DataFrame(rows)


def select_representative_model(pre_para, similarity_df=None, models=tuple(MODELS), save_dir=None):
    """
    Combine both axes and print a recommendation:
      - representativeness: model_representativeness_score() -- most
        "typical" curve shape, from model_curve_similarity data
      - performance: model_performance_score() -- best actual results

    If the same model tops both rankings, it's a clean pick. If they
    disagree, both tables are returned so the conflict can be reported
    explicitly rather than picked silently.
    """
    if similarity_df is None:
        similarity_df = analyze_curve_similarity(pre_para, models=models, save_dir=save_dir)

    rep_df = model_representativeness_score(similarity_df, models=models)
    perf_df = model_performance_score(pre_para, models=models, save_dir=save_dir)

    most_representative = rep_df.iloc[0]["model"]
    best_macro_f1 = perf_df.loc[perf_df["mean_macro_f1"].idxmax(), "model"]

    print(f"most representative model (closest to the other two on average): {most_representative}")
    print(f"best-performing model (mean macro_f1 across the sweep): {best_macro_f1}")
    if most_representative == best_macro_f1:
        print(f"-> {most_representative} tops both rankings: a clean, well-supported pick.")
    else:
        print("-> rankings disagree: report this conflict explicitly rather than "
              "picking silently (see rep_df / perf_df).")

    return rep_df, perf_df


def render_curve_similarity_image(similarity_df, out_path: str, figure_title: str) -> str:
    display_cols = ["image_title", "curve", "model_a", "model_b", "n_matched",
                     "pearson_r", "spearman_r", "nrmse"]
    display_cols = [c for c in display_cols if c in similarity_df.columns]
    rows = [display_cols] + [
        [_fmt(row[c]) for c in display_cols]
        for _, row in similarity_df.sort_values("spearman_r").iterrows()
    ]
    return render_table_image(rows, out_path, figure_title, col_width=250, row_height=34, font_size=15)


def _group_report_dir(pre_para):
    """Same out_dir run_compose_report() writes model_curve_similarity.csv
    into, for one experimental group."""
    return os.path.join(
        REPORT_OUTPUT_DIR,
        f"{pre_para.symbol}_{pre_para.interval}",
        pre_para.label_type,
        pre_para.para_type,
    )


CURVE_SIMILARITY_REPORT_DIR = os.path.join(common.OUTPUT_DIR, "model_curve_similarity_report")


def aggregate_curve_similarity(paras=None, models=tuple(MODELS), save_dir=None, report_dir=None):
    """Run analyze_curve_similarity() over every experimental group (default:
    all 12 groups from the dissertation, via crossing_specificity.EXPERIMENTAL_GROUPS)
    and write exactly 2 consolidated CSVs, plus print the overall weighted
    consistency score across every group/curve/model pair:

        model_curve_similarity_detail.csv   - every group x curve x model pair
        model_curve_similarity_summary.csv  - averaged across groups, one row
            per (image_title, curve, model_a, model_b), sorted by
            mean_spearman_r ascending so the LEAST cross-model-consistent
            curves are at the top. Includes min_spearman_r (the worst
            single group) so a curve that looks fine on average but fails
            badly in one group doesn't get missed.

    Reuses each group's model_curve_similarity.csv if run_compose_report()
    already produced one (no need to recompute) — only groups that haven't
    been composed yet get analyze_curve_similarity() called fresh here.
    """
    paras = paras or crossing_specificity.EXPERIMENTAL_GROUPS
    report_dir = report_dir or CURVE_SIMILARITY_REPORT_DIR
    os.makedirs(report_dir, exist_ok=True)

    all_details = []
    for para in paras:
        cached_path = os.path.join(_group_report_dir(para), "model_curve_similarity.csv")
        try:
            if os.path.exists(cached_path):
                d = pd.read_csv(cached_path)
            else:
                d = analyze_curve_similarity(para, models=models, save_dir=save_dir)
            all_details.append(d)
        except Exception as e:
            print(f"[skip group] {para.symbol}_{para.interval}/"
                  f"{para.label_type}/{para.para_type}: {e}")

    if not all_details:
        raise RuntimeError("No experimental group produced results — check output/ paths.")

    detail = pd.concat(all_details, ignore_index=True)
    summary = (
        detail.groupby(["image_title", "curve", "model_a", "model_b"])
        .agg(
            n_groups=("spearman_r", "size"),
            mean_pearson_r=("pearson_r", "mean"),
            mean_spearman_r=("spearman_r", "mean"),
            min_spearman_r=("spearman_r", "min"),
            mean_nrmse=("nrmse", "mean"),
            max_nrmse=("nrmse", "max"),
        )
        .reset_index()
        .sort_values("mean_spearman_r")
    )

    detail_path = os.path.join(report_dir, "model_curve_similarity_detail.csv")
    summary_path = os.path.join(report_dir, "model_curve_similarity_summary.csv")
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"detail (all groups) saved: {detail_path}")
    print(f"summary (all groups) saved: {summary_path}")

    score = overall_similarity_score(detail)
    print(f"overall weighted cross-model consistency score (spearman_r): {score:.3f}")

    return detail, summary, score


# ======================================================================
# Main entry point
# ======================================================================

def compose_distribution_overview_grid(
    paras,
    out_path: str = None,
    n_cols: int = 2,
    thumb_width: int = THUMB_WIDTH,
    figure_title: str = "distribution_overview",
) -> str:
    """Collect regime_discovery_output/.../distribution_overview.png for every
    config in `paras` and stitch them into one grid image, 2 per row."""
    images = []
    for pre_para in paras:
        img_path = os.path.join(resolve_base_dir(pre_para, "regime"), "distribution_overview.png")
        if os.path.isfile(img_path):
            title = f"{pre_para.symbol}_{pre_para.interval}_{pre_para.label_type}_{pre_para.para_type}"
            images.append((title, img_path))
        else:
            print(f"[skip] image not found: {img_path}")

    if not images:
        raise RuntimeError("No distribution_overview.png found for the given paras.")

    canvas = compose_grid(images, n_cols=n_cols, thumb_width=thumb_width, figure_title=figure_title)

    if out_path is None:
        os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(REPORT_OUTPUT_DIR, "distribution_overview_grid.png")
    canvas.save(out_path)
    print(f"saved: {out_path}")
    return out_path


def run_compose_report(pre_para):
    # Mirror train.py's save_dir layout: <REPORT_OUTPUT_DIR>/{symbol}_{interval}/{label_type}/{para_type}/
    out_dir = os.path.join(
        REPORT_OUTPUT_DIR,
        f"{pre_para.symbol}_{pre_para.interval}",
        pre_para.label_type,
        pre_para.para_type,
    )
    os.makedirs(out_dir, exist_ok=True)
    base_title = f"{pre_para.symbol}_{pre_para.interval}_{pre_para.label_type}_{pre_para.para_type}"

    for model in MODELS:
        images = collect_images(pre_para, model)
        if not images:
            print(f"[skip] no images found for {base_title}/{model}")
            continue
        canvas = compose_grid(images, n_cols=N_COLS, thumb_width=THUMB_WIDTH,
                               figure_title=f"{base_title}_{model}")
        out_path = os.path.join(out_dir, f"{model}_report.png")
        canvas.save(out_path)
        print(f"saved: {out_path}")

    # --- crossing-point specificity (delegates the test statistics to
    # crossing_specificity.py; this module only renders the result) -------
    try:
        master_df, consistency_df = crossing_specificity.run_full_analysis(
            pre_para, models=tuple(MODELS),
        )
    except Exception as e:
        print(f"[skip] crossing specificity analysis failed for {base_title}: {e}")
    else:
        table_path = os.path.join(out_dir, "crossing_specificity_summary.png")
        render_consistency_table_image(consistency_df, table_path, f"{base_title}_crossing_specificity")
        print(f"saved: {table_path}")

    # --- cross-model curve similarity -------------------------------------
    try:
        similarity_df = analyze_curve_similarity(pre_para, models=tuple(MODELS))
    except Exception as e:
        print(f"[skip] model curve similarity failed for {base_title}: {e}")
    else:
        csv_path = os.path.join(out_dir, "model_curve_similarity.csv")
        similarity_df.to_csv(csv_path, index=False)
        img_path = os.path.join(out_dir, "model_curve_similarity.png")
        render_curve_similarity_image(similarity_df, img_path, f"{base_title}_model_curve_similarity")
        print(f"saved: {csv_path}")
        print(f"saved: {img_path}")
        score = overall_similarity_score(similarity_df)
        print(f"[{base_title}] overall weighted cross-model consistency score (spearman_r): {score:.3f}")


if __name__ == "__main__":
    # Per-config image composition (one call per BaseDefine you want a
    # composed report for):
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
    # compose_distribution_overview_grid(paras)
    for pre_para in paras:
        run_compose_report(pre_para)

    # Cross-group consistency check (all 12 dissertation groups pooled into
    # output/model_curve_similarity_report/, 2 CSVs + one printed overall
    # score): run this separately once every group has been through
    # run_compose_report at least once.
    detail_all, summary_all, overall_score = aggregate_curve_similarity(paras = paras)