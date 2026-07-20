#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_run.py
------------
Sequentially runs, for every config dict in `CONFIGS` (or whatever list you
pass to run_batch()):

    1. preparation.py main(mode='batch_label') -- sweeps every threshold,
       attaches all label_v/label_h columns, saves the labeled train/test
       split + meta.json.
    2. preparation.py main(mode='plot')        -- sweeps the label-ratio-
       vs-Gaussian curves and saves the crossing-point CSVs that
       crossing_specificity.py depends on.
    3. train.py main()                          -- trains all 3 models
       across every threshold x 50 runs, writes every summary CSV, and
       calls compose_report_images.run_compose_report() at the end.

Each config's arguments are a plain kwargs dict with EXACTLY the same
keys/values as the corresponding `main(logger, args, common.FEATURE_GROUP_LIST,
para=..., volatility_range=..., predict_num=...)` call commented out at the
bottom of preparation.py's __main__ -- just the `para=..., ...` tail wrapped
in `dict(...)`. Keeping the two lists in this same shape means updating one
is a straight copy-paste into the other, no reformatting.

Each config gets its OWN prep_output_dir (a subfolder of common.DATA_OUT_DIR
named after symbol_interval_labeltype_paratype), since preparation.py's
default prep_output_dir is a single shared location -- running two configs
back-to-back without this would silently overwrite the first config's
labeled data before train.py ever reads it.

train.py's own training loop is already parallelized internally
(ProcessPoolExecutor over the 50 repeated runs), so this script does NOT
add another layer of parallelism across configs -- configs run one after
another.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
sys.path.append(current_work_dir)

from data_process import common
from data_process import preparation
from model.train_config import TrainConfig
import train as train_module


# ----------------------------------------------------------------------
# One dict per preparation.py main() call. Keep this list in the SAME
# ORDER and SAME ARGUMENT SHAPE as the commented-out calls at the bottom of
# preparation.py's __main__ -- to update either file, copy the `para=...,
# ...` tail of a main(...) call and wrap it in dict(...), or vice versa.
# Comment out an entry here the same way preparation.py comments out a line
# to skip a config.
# ----------------------------------------------------------------------

def _prep_output_dir(para):
    """Per-config isolation for preparation.py's output -- prevents the
    next config's batch_label run from overwriting this one's labeled
    train/test data before train.py has read it."""
    return os.path.join(
        common.DATA_OUT_DIR,
        f"{para.symbol}_{para.interval}_{para.label_type}_{para.para_type}",
    )


def run_one(logger, config: dict, train_cfg=None):
    """Run preparation.py (batch_label + plot) then train.py's main() for
    ONE config. `config` is a kwargs dict exactly like one entry of
    CONFIGS: {"para": ..., "volatility_range"/"horizon_range": ...,
    "predict_num"/"vol_multiplier": ...}."""
    para = config["para"]
    prep_output_dir = _prep_output_dir(para)
    os.makedirs(prep_output_dir, exist_ok=True)
    logger.info(
        f"=== {para.symbol}_{para.interval}/{para.label_type}/{para.para_type} "
        f"-> prep_output_dir={prep_output_dir} ==="
    )

    range_kwargs = {k: v for k, v in config.items()
                     if k in ("volatility_range", "horizon_range")}
    scalar_kwargs = {k: v for k, v in config.items()
                      if k in ("predict_num", "vol_multiplier")}

    # 1) batch_label: sweep every threshold, attach all label_v/label_h
    #    columns, save train/test split + meta.json into prep_output_dir.
    t0 = time.time()
    preparation.main(
        logger, argparse.Namespace(mode="batch_label"),
        common.FEATURE_GROUP_LIST, para=para, prep_output_dir=prep_output_dir,
        **range_kwargs, **scalar_kwargs,
    )
    logger.info(f"batch_label done in {time.time() - t0:.1f}s")

    # 2) plot: only needs the fixed scalar -- preparation.py's own
    #    plot_label_distribution() sweeps the full parameter_range
    #    internally, independent of range_kwargs above. Reads the raw CSV
    #    directly (not prep_output_dir), so this is independent of step 1;
    #    it writes into output/regime_discovery_output/{symbol}_{interval}/
    #    {label_type}/{para_type}/, which is already isolated per config.
    # t0 = time.time()
    # preparation.main(
    #     logger, argparse.Namespace(mode="plot"),
    #     common.FEATURE_GROUP_LIST, para=para, prep_output_dir=prep_output_dir,
    #     **scalar_kwargs,
    # )
    # logger.info(f"plot done in {time.time() - t0:.1f}s")

    # 3) train: loads the labeled df from prep_output_dir, trains all 3
    #    models across every threshold x 50 runs (ProcessPoolExecutor
    #    internally, hardcoded to 25 workers inside train.main() -- not
    #    something this script controls), writes everything under
    #    common.TRAIN_OUT_DIR/{symbol}_{interval}/{label_type}/{para_type}/,
    #    and calls compose_report_images.run_compose_report() at the end.
    t0 = time.time()
    train_module.main(
        logger, train_cfg=train_cfg or TrainConfig(), pre_para=para,
        prep_output_dir=prep_output_dir, save_dir=common.TRAIN_OUT_DIR,
    )
    logger.info(f"train done in {time.time() - t0:.1f}s")


def run_batch(configs=None, train_cfg=None, run_aggregation=True):
    """
    Sequentially run_one() for every config in `configs` (default: CONFIGS
    above) -- ONE AT A TIME. train.py's own ProcessPoolExecutor already
    parallelizes the 50 runs within a single config, so this loop stays a
    plain for-loop; running multiple configs concurrently would
    oversubscribe the same CPU/GPU that pool is already using.

    A failure in one config is logged and skipped (not raised) so the rest
    of the batch still runs -- check the log for FAILED entries afterward.

    run_aggregation: if True, once every config has finished, also calls
    crossing_specificity.run_all_groups() and
    compose_report_images.aggregate_curve_similarity() to produce the
    cross-group summary CSVs covering everything just run.
    """
    logger, _ = common.setup_session_logger(sub_folder="batch_run", file_level=logging.DEBUG)

    failed = []
    for i, config in enumerate(configs):
        para = config["para"]
        key = (para.symbol, para.interval, para.label_type, para.para_type)
        logger.info(f"[{i + 1}/{len(configs)}] starting {key}")
        begin = time.time()
        try:
            run_one(logger, config, train_cfg=train_cfg)
        except Exception:
            logger.exception(f"[{i + 1}/{len(configs)}] {key} FAILED -- continuing with remaining configs")
            failed.append(key)
            continue
        logger.info(f"[{i + 1}/{len(configs)}] {key} finished in {time.time() - begin:.1f}s")

    if failed:
        logger.warning(f"batch finished with {len(failed)} failed config(s): {failed}")
    else:
        logger.info("batch finished, all configs succeeded")

    if run_aggregation:
        from analysis import crossing_specificity
        from analysis import compose_report_images
        paras = [c["para"] for c in configs]
        try:
            crossing_specificity.run_all_groups(paras=paras)
        except Exception:
            logger.exception("crossing_specificity.run_all_groups() failed")
        try:
            compose_report_images.aggregate_curve_similarity(paras=paras)
        except Exception:
            logger.exception("compose_report_images.aggregate_curve_similarity() failed")

    return failed


if __name__ == "__main__":
    groups = [
        dict(para=common.BTC_15m_fthl_volatility,     volatility_range=np.arange(0.1, 10.1, 0.1).round(1), predict_num=16),
        dict(para=common.BTC_15m_fthl_horizon,        horizon_range=np.arange(16, 81, 1),                   vol_multiplier=10),
        dict(para=common.BTC_15m_tbm_volatility,      volatility_range=np.arange(1.4, 10.1, 0.1).round(1),  predict_num=16),
        dict(para=common.BTC_15m_tbm_horizon,         horizon_range=np.arange(8, 81, 1),                    vol_multiplier=10),

        dict(para=common.XAUUSD_15m_fthl_volatility,  volatility_range=np.arange(0.2, 15.1, 0.1).round(1),  predict_num=16),
        dict(para=common.XAUUSD_15m_fthl_horizon,     horizon_range=np.arange(16, 81, 1),                   vol_multiplier=10),
        dict(para=common.XAUUSD_15m_tbm_volatility,   volatility_range=np.arange(1.2, 15.1, 0.1),           predict_num=16),  # 注：此行原来就没有 .round(1)
        dict(para=common.XAUUSD_15m_tbm_horizon,      horizon_range=np.arange(8, 81, 1),                    vol_multiplier=10),

        dict(para=common.XAUUSD_1d_fthl_volatility,   volatility_range=np.arange(0.4, 4.1, 0.1).round(1),   predict_num=8),
        dict(para=common.XAUUSD_1d_fthl_horizon,      horizon_range=np.arange(16, 41, 1),                   vol_multiplier=4),
        dict(para=common.XAUUSD_1d_tbm_volatility,    volatility_range=np.arange(2, 4.1, 0.1).round(1),     predict_num=8),
        dict(para=common.XAUUSD_1d_tbm_horizon,       horizon_range=np.arange(8, 41, 1),                    vol_multiplier=4),
    ]
    run_batch(configs= groups)