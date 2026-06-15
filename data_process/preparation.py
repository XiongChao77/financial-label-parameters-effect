from pickle import FALSE
from tkinter import TRUE
import pandas as pd
import numpy as np
import datetime, os, sys, re, math, json, logging, argparse
from dataclasses import asdict
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
from data_process import common

def plot_label_distribution(df, interval_ms, para, label_function):
    from data_process.plot_label_ratio_sweep import LabelRatioCurveAnalyzer

    analyzer = LabelRatioCurveAnalyzer(df, interval_ms, para)

    stop_range = [np.inf]

    if para.para_type == 'volatility':
        parameter_range = np.arange(0.1, 5, 0.1).round(2)
    elif para.para_type == 'horizon':
        parameter_range = np.arange(2, 80, 2).astype(int)
    else:
        raise ValueError(f"Unsupported para_type: {para.para_type}")

    analyzer.run_parameter_sweep(
        parameter_range=parameter_range,
        stop_range=stop_range,
        fun=label_function,
        label_col='label',
    )

    analyzer.plot_all_label_ratio_curves(
        stop_rate=np.inf,
        smooth_window=3,
        include_gaussian=True,
    )

def summary_statistics_stock(para, prep_output_dir, df, label_col, logger):
    """
    针对股票日线数据优化的统计与保存函数
    适配格式: date, open, high, low, close, volume
    """
    # 1. 确保日期格式正确
    df['date'] = pd.to_datetime(df['date'])
    
    # ---------------- 基础时间跨度统计 ----------------
    start_time = df['date'].iloc[0]
    end_time = df['date'].iloc[-1]
    duration = end_time - start_time
    logger.info(f"📊 数据时间跨度: {start_time.date()} -> {end_time.date()} (总计: {duration.days} 天)")

    # ---------------- 标签分布统计 ----------------
    counts = df[label_col].value_counts().sort_index()
    proportions = df[label_col].value_counts(normalize=True).sort_index()
    
    logger.info("\n=== 标签分布汇总 (Label Distribution) ===")
    label_map = {0: "Down (Short)", 1: "Neutral", 2: "Up (Long)", -1: "INVALID"}
    
    for label_val, cnt in counts.items():
        name = label_map.get(label_val, "Unknown")
        pct = proportions[label_val]
        logger.info(f"Label {label_val:>2} ({name:<12}): {cnt:>6} 行, 占比 {pct:.2%}")

    # ---------------- 动态阈值分析 ----------------
    if 'threshold_long' in df.columns and 'threshold_short' in df.columns:
        logger.info("\n=== 动态波动率阈值分析 (Dynamic Thresholds) ===")
        for side in ['long', 'short']:
            col = f'threshold_{side}'
            logger.info(f"{side.capitalize():<5} Threshold | Min: {df[col].min():.4%}, Max: {df[col].max():.4%}, Mean: {df[col].mean():.4%}")
    
    # ---------------- 数据切分 (Train/Test Split) ----------------
    # 沿用你代码中 8 个月的切分逻辑，如果是多年日线数据，可以根据需要调整为 20%
    split_date = df['date'].max() - pd.DateOffset(months=8)
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()

    logger.info(f"\n📂 数据集切分结果:")
    logger.info(f"   - 训练集: {len(train_df)} 行 (截至 {train_df['date'].max().date()})")
    logger.info(f"   - 测试集: {len(test_df)} 行 (始于 {test_df['date'].min().date()})")

    # ---------------- 保存数据与元数据 ----------------
    os.makedirs(prep_output_dir, exist_ok=True)
    
    # 调用你 common 中的保存方法
    common.save_train_df_to_dir(train_df, prep_output_dir)
    common.save_test_df_to_dir(test_df, prep_output_dir)
    
    # 保存配置 JSON
    meta_path = common.get_data_config_path_in_dir(prep_output_dir)
    para_dict = asdict(para)
    # 确保 JSON 中没有无法序列化的对象
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(common.json_safe(para_dict), f, indent=4, ensure_ascii=False)

    logger.info(f"✅ 数据准备完成。配置文件已写入: {meta_path}")

def summary_statistics(para,prep_output_dir,df, logger):
    label_cols = [
        col for col in df.columns 
        if col.startswith('label_') and not any(sub in col for sub in ['index', 'id', 'exit', 'price', 'threshold', 'strength', 'stop'])
    ]
    
    if label_cols:
        # 此时剩下的全部是真正的标签列（如 label_3.5_3.5），再从中安全地寻找 0 最少的列
        label_col = min(label_cols, key=lambda c: int((df[c] == 0).sum()))
        logger.info(f"strictest label column is {label_col}")

    # ---------------- Summary statistics ----------------
    start_time = df['open_time_date_utc'].iloc[0]
    end_time = df['open_time_date_utc'].iloc[-1]
    duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
    logger.info(f"Time span: {start_time} -> {end_time} (total: {duration})")
    counts = df[label_col].value_counts().sort_index()
    proportions = df[label_col].value_counts(normalize=True).sort_index()
    
    logger.info("\n=== Dynamic label distribution summary ===")
    logger.info("Thresholds are saved in columns: 'threshold_long' and 'threshold_short'")
    logger.info(f"Long threshold range: Min={df['threshold_long'].min():.4f}, Max={df['threshold_long'].max():.4f}, Mean={df['threshold_long'].mean():.4f}")
    logger.info(f"Short threshold range: Min={df['threshold_short'].min():.4f}, Max={df['threshold_short'].max():.4f}, Mean={df['threshold_short'].mean():.4f}")
    
    for label_val, cnt in counts.items():
        label_name = "Down" if label_val == 0 else ("Up" if label_val == 2 else ("Neutral" if label_val == 1 else "INVALID"))
        pct_val = proportions[label_val]
        logger.info(f"Label {label_val} ({label_name}): {cnt} rows, ratio {pct_val:.4%}")
    logger.info("==========================\n")
    
    # ---------------------------------------------------------
    # 3. Split and persist datasets
    # ---------------------------------------------------------
    split_ts = pd.to_datetime(df['open_time_date_utc'].iloc[-1]) - pd.DateOffset(months=8)
    train_df, test_df = df[df['open_time_date_utc'] < str(split_ts)], df[df['open_time_date_utc'] >= str(split_ts)]

    start_time = train_df['open_time_date_utc'].iloc[0]
    end_time = train_df['open_time_date_utc'].iloc[-1]
    duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
    logger.info(f"Train Time span: {start_time} -> {end_time} (total: {duration})")

    counts = train_df[label_col].value_counts().sort_index()
    proportions = train_df[label_col].value_counts(normalize=True).sort_index()
    for label_val, cnt in counts.items():
        label_name = "Down" if label_val == 0 else ("Up" if label_val == 2 else ("Neutral" if label_val == 1 else "INVALID"))
        pct_val = proportions[label_val]
        logger.info(f"Label {label_val} ({label_name}): {cnt} rows, ratio {pct_val:.4%}")

    # Write to prep_output_dir (default common.DATA_OUT_DIR; independent per batch worker)
    out_dir = prep_output_dir
    os.makedirs(out_dir, exist_ok=True)
    common.save_train_df_to_dir(train_df, out_dir)
    common.save_test_df_to_dir(test_df, out_dir)
    meta_path = common.get_data_config_path_in_dir(out_dir)
    para_dict = asdict(para)
    safe_para = common.json_safe(para_dict)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(safe_para, f, indent=4, ensure_ascii=False)

    logger.info("✅ Data preparation completed.")
    logger.info(f"📍 Interval: {para.interval}")
    logger.info(f"📍 Config written to: {meta_path}")

def main(logger:logging.Logger, args, feature_group_list = common.FEATURE_GROUP_LIST,feature_conf_list=[],para = common.BaseDefine(), prep_output_dir =common.DATA_OUT_DIR,**kwargs ):
    file = os.path.join(common.PROJECT_DATA_DIR,para.market_category, para.data_source ,para.trading_type ,f"{para.symbol}_{para.interval}.csv")
    logger.info(f"using file :{file}")
    # 1. Convert interval string to milliseconds
    interval_ms = common.get_interval_ms(para.interval)
    
    # 2. Persist metadata for labeling and downstream model usage
    df = pd.read_csv(file)
    # Rows with volume==0 typically have little impact on price, so dropping them often won't hurt training/testing.
    # In real-world feeds, volume==0 can exist; we keep them by default.
    # Feature engineering must explicitly handle volume==0 cases.
    df = common.clean_data_quality_auto(df,logger)
    # 3. Pass interval_ms to label logic so it can adapt its volatility window to the real time span.
    if args.mode == 'plot':
        plot_label_distribution(df, interval_ms, para, common.attach_label)        # 4. Run analysis

    elif args.mode == 'batch_label':
        def generate_strict_consensus_label(df, label_prefix='label_v'):
            """
            Dissertation logic: extremely strict label consensus.
            - All columns Positive -> Signal.POSITIVE (2)
            - All columns Negative -> Signal.NEGATIVE (0)
            - All columns Neutral  -> Signal.NEUTRAL (1)
            - Otherwise (mixed directions or trend-to-range transitions) -> Signal.INVALID (-1)
            """
            label_col = 'label'
            label_cols = [c for c in df.columns if c.startswith(label_prefix)]
            if not label_cols:
                return df

            # Check whether all label columns agree on each row.
            # .nunique(axis=1) == 1 means all label_vXX columns point to the same outcome.
            is_unanimous = df[label_cols].nunique(axis=1) == 1
            df[label_col] = common.Signal.INVALID
            
            # Only rows with full consensus inherit the shared label value (0, 1, or 2).
            df.loc[is_unanimous, label_col] = df.loc[is_unanimous, label_cols[0]]
            
            return df

        if para.para_type == 'volatility':
            para.predict_num = kwargs['predict_num']
            volatility_range = kwargs['volatility_range']
            for t_range in volatility_range:
                para.vol_multiplier_long = t_range
                para.stop_multiplier_rate_long = None
                para.vol_multiplier_short = t_range
                para.stop_multiplier_rate_short = None
                label_suffix = f"v{int(round(t_range * 10)):02d}"
                para_label = f'label_{label_suffix}'
                df = common.attach_label(df, para=para, label_col=para_label)
        elif para.para_type == 'horizon':
            vol_multiplier = kwargs['vol_multiplier']
            para.vol_multiplier_long = vol_multiplier
            para.stop_multiplier_rate_long = None
            para.vol_multiplier_short = vol_multiplier
            para.stop_multiplier_rate_short = None
            horizon_range = kwargs['horizon_range']
            for t_range in horizon_range:
                para.predict_num = t_range
                label_suffix = f"h{int(round(t_range * 10)):02d}"
                para_label = f'label_{label_suffix}'
                df = common.attach_label(df, para=para, label_col=para_label)
        df = generate_strict_consensus_label(df)
        # ---------------- Summary statistics ----------------
        summary_statistics(para, prep_output_dir, df, logger)
    elif args.mode == 'label':
        df = common.attach_label(df, para=para, label_col=f'label')
        summary_statistics(para, prep_output_dir, df, logger)
    else:
        logger.warning("No action specified. Use --plot to visualize label distribution or --label to generate labels and summary statistics.")

if __name__ == "__main__":
#**********column info: open_time_date_utc,open,high,low,close,volume,close_time_ms_utc,quote_asset_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,ignore
    parser = argparse.ArgumentParser(description="Dissertation data preparation script with dynamic labeling and feature engineering.")
    parser.add_argument(
        "--mode", 
        choices=["plot", "label", "batch_label"], 
        default="batch_label",
        help="Action to perform: 'plot' the distribution or 'label' the data and run stats (default: label)")
    
    args = parser.parse_args()
    logger, _ = common.setup_session_logger(sub_folder='dissertation/data_process')
    
    # main(logger, args, common.FEATURE_GROUP_LIST, para = common.BTC_15m_fthl_volatility,volatility_range= np.arange(0.5, 10, 0.1).round(1),predict_num=16) 
    # main(logger, args, common.FEATURE_GROUP_LIST, para = common.BTC_15m_fthl_horizon, horizon_range= np.arange(2, 80, 2), vol_multiplier = 4)
    # main(logger, args, common.FEATURE_GROUP_LIST, para = common.BTC_15m_tbm_volatility, volatility_range= np.arange(0.5, 8, 0.1).round(1), predict_num= 16)
    # main(logger, args, common.FEATURE_GROUP_LIST, para = common.BTC_15m_tbm_horizon, horizon_range =  np.arange(2, 80, 2), vol_multiplier = 4)
    # main(logger, args, common.FEATURE_GROUP_LIST, para = common.QQQ_15m_fthl_volatility, volatility_range= np.arange(0.1, 6, 0.1).round(1), predict_num = 16)
    # main(logger, args, common.FEATURE_GROUP_LIST, para = common.QQQ_15m_fthl_horizon, horizon_range= np.arange(2, 24, 2), vol_multiplier=3)
    # main(logger, args, common.FEATURE_GROUP_LIST, para = common.QQQ_15m_tbm_volatility, volatility_range= np.arange(0.1, 6, 0.1), predict_num= 16)
    # main(logger, args, common.FEATURE_GROUP_LIST, para = common.QQQ_15m_tbm_horizon, horizon_range= np.arange(2, 36, 2), vol_multiplier=4)
    # main(logger, args, common.FEATURE_GROUP_LIST, para = common.QQQ_1d_fthl_volatility, volatility_range= np.arange(0.1, 4, 0.1).round(1), predict_num= 16)
    # main(logger, args, common.FEATURE_GROUP_LIST, para = common.QQQ_1d_fthl_horizon, horizon_range= np.arange(2, 80, 2), vol_multiplier=2)
    # main(logger, args, common.FEATURE_GROUP_LIST, para = common.QQQ_1d_tbm_volatility, volatility_range= np.arange(0.5, 3, 0.1).round(1), predict_num= 16)
    main(logger, args, common.FEATURE_GROUP_LIST, para = common.QQQ_1d_tbm_horizon, horizon_range= np.arange(2, 16, 2), vol_multiplier= 2)