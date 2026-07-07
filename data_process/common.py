from enum import IntEnum,Enum
from functools import lru_cache
from dataclasses import dataclass
import logging,math,re,git
import pandas as pd
import numpy as np
import os, colorlog , logging, json,platform
from dataclasses import asdict, is_dataclass,fields
from typing import Optional
from datetime import datetime
from data_process.utils import *
from data_process.feature import *
from numba import njit

class Signal(IntEnum):
    INVALID = -1
    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE  = 2

eps = 1e-8

DATA_PROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DATA_PROCESS_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_OUT_DIR = os.path.join(OUTPUT_DIR, "data")
os.makedirs(DATA_OUT_DIR, exist_ok=True)

@dataclass
class BaseDefine:
    #data source
    market_category: str = "Cryptocurrency"   # cryptocurrency / Stock / Forex
    data_source: str = "binance_public_data"                   # binance / yahoo / dukascopy
    # model
    vol_ewma_span: int  = 80
    predict_num: int = 16
    # risk / vol
    vol_multiplier_long: float = 4
    stop_multiplier_rate_long: Optional[float] = None
    vol_multiplier_short: float = 4
    stop_multiplier_rate_short: Optional[float] = None
    # market
    symbol: str = "BTCUSDT"    #BTCUSDT ETHUSDT DOGEUSDT XAUUSD
    trading_type:str ='spot'             #spot  / um(USDT-M Futures) / cm    (Coin-M Futures)
    interval: str = "15m"
    para_type:str =  'horizon'  # volatility / horizon
    label_type:str = 'FTHL' # TBM / FTHL
    version:float = 0.1

BTC_15m_fthl_volatility = BaseDefine(market_category="Cryptocurrency", data_source="binance_public_data", symbol="BTCUSDT", interval="15m", trading_type='spot', para_type = 'volatility', label_type = 'FTHL')
BTC_15m_fthl_horizon = BaseDefine(market_category="Cryptocurrency", data_source="binance_public_data", symbol="BTCUSDT", interval="15m", trading_type='spot', para_type = 'horizon', label_type = 'FTHL')
BTC_15m_tbm_volatility = BaseDefine(market_category="Cryptocurrency", data_source="binance_public_data", symbol="BTCUSDT", interval="15m", trading_type='spot', para_type = 'volatility', label_type = 'TBM')
BTC_15m_tbm_horizon = BaseDefine(market_category="Cryptocurrency", data_source="binance_public_data", symbol="BTCUSDT", interval="15m", trading_type='spot', para_type = 'horizon', label_type = 'TBM')

QQQ_15m_fthl_volatility = BaseDefine(market_category="Stock", data_source="massive", symbol="QQQ", interval="15m", trading_type='spot', version=0.1, para_type = 'volatility', label_type = 'FTHL')
QQQ_15m_fthl_horizon = BaseDefine(market_category="Stock", data_source="massive", symbol="QQQ", interval="15m", trading_type='spot', version=0.1, para_type = 'horizon', label_type = 'FTHL')
QQQ_15m_tbm_volatility = BaseDefine(market_category="Stock", data_source="massive", symbol="QQQ", interval="15m", trading_type='spot', version=0.1, para_type = 'volatility', label_type = 'TBM')
QQQ_15m_tbm_horizon = BaseDefine(market_category="Stock", data_source="massive", symbol="QQQ", interval="15m", trading_type='spot', version=0.1, para_type = 'horizon', label_type = 'TBM')

QQQ_1d_fthl_volatility = BaseDefine(market_category="Stock", data_source="massive", symbol="QQQ", interval="1d", trading_type='spot', version=0.1, para_type = 'volatility', label_type = 'FTHL')
QQQ_1d_fthl_horizon = BaseDefine(market_category="Stock", data_source="massive", symbol="QQQ", interval="1d", trading_type='spot', version=0.1, para_type = 'horizon', label_type = 'FTHL')
QQQ_1d_tbm_volatility = BaseDefine(market_category="Stock", data_source="massive", symbol="QQQ", interval="1d", trading_type='spot', version=0.1, para_type = 'volatility', label_type = 'TBM')
QQQ_1d_tbm_horizon = BaseDefine(market_category="Stock", data_source="massive", symbol="QQQ", interval="1d", trading_type='spot', version=0.1, para_type = 'horizon', label_type = 'TBM')

XAUUSD_15m_fthl_volatility = BaseDefine(market_category="Forex", data_source="dukascopy", symbol="XAUUSD", interval="15m", trading_type='spot', version=0.1, para_type = 'volatility', label_type = 'FTHL')
XAUUSD_15m_fthl_horizon = BaseDefine(market_category="Forex", data_source="dukascopy", symbol="XAUUSD", interval="15m", trading_type='spot', version=0.1, para_type = 'horizon', label_type = 'FTHL')
XAUUSD_15m_tbm_volatility = BaseDefine(market_category="Forex", data_source="dukascopy", symbol="XAUUSD", interval="15m", trading_type='spot', version=0.1, para_type = 'volatility', label_type = 'TBM')
XAUUSD_15m_tbm_horizon = BaseDefine(market_category="Forex", data_source="dukascopy", symbol="XAUUSD", interval="15m", trading_type='spot', version=0.1, para_type = 'horizon', label_type = 'TBM')

XAUUSD_1d_fthl_volatility = BaseDefine(market_category="Forex", data_source="dukascopy", symbol="XAUUSD", interval="1d", trading_type='spot', version=0.1, para_type = 'volatility', label_type = 'FTHL')
XAUUSD_1d_fthl_horizon = BaseDefine(market_category="Forex", data_source="dukascopy", symbol="XAUUSD", interval="1d", trading_type='spot', version=0.1, para_type = 'horizon', label_type = 'FTHL')
XAUUSD_1d_tbm_volatility = BaseDefine(market_category="Forex", data_source="dukascopy", symbol="XAUUSD", interval="1d", trading_type='spot', version=0.1, para_type = 'volatility', label_type = 'TBM')
XAUUSD_1d_tbm_horizon = BaseDefine(market_category="Forex", data_source="dukascopy", symbol="XAUUSD", interval="1d", trading_type='spot', version=0.1, para_type = 'horizon', label_type = 'TBM')


# SPY_1d_fthl_volatility = BaseDefine(market_category="Stock", data_source="massive", symbol="SPY", interval="1d", trading_type='spot', version=0.1, para_type = 'horizon')

log_level = logging.INFO

PROJECT_DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR),'QuantData')
train_data_path = os.path.join(DATA_OUT_DIR, "train_data.csv")
test_data_path  = os.path.join(DATA_OUT_DIR, "test_data.csv")
data_config_path  = os.path.join(DATA_OUT_DIR, "data_config_meta.json")
TRAIN_OUT_DIR = os.path.join(OUTPUT_DIR, "train")
os.makedirs(TRAIN_OUT_DIR, exist_ok=True)

CONF_DF = 'to_feather'#/'to_feather'/'to_csv'

def save_train_df(df):
    if os.path.exists(train_data_path):
        os.remove(train_data_path)
    if CONF_DF == 'to_csv':
        df.to_csv(train_data_path, index=False, encoding="utf-8")
    else:
        df.columns = df.columns.astype(str)
        df.to_feather(train_data_path)


def load_train_df():
    if CONF_DF == 'to_csv':
        return pd.read_csv(train_data_path, encoding="utf-8")
    else:
        return pd.read_feather(train_data_path)

def save_test_df(df):
    if os.path.exists(test_data_path):
        os.remove(test_data_path)
    if CONF_DF == 'to_csv':
        df.to_csv(test_data_path, index=False, encoding="utf-8")
    else:
        df.columns = df.columns.astype(str)
        df.to_feather(test_data_path)

def load_test_df():
    if CONF_DF == 'to_csv':
        return pd.read_csv(test_data_path, encoding="utf-8")
    else:
        return pd.read_feather(test_data_path)

# ---------- Per-directory read/write (for batch multiprocessing: each preparation uses its own directory) ----------
def _data_path_in_dir(base_dir, name):
    return os.path.join(base_dir, name)

def save_train_df_to_dir(df, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    path = _data_path_in_dir(base_dir, "train_data.csv" if CONF_DF == 'to_csv' else "train_data.feather")
    if os.path.exists(path):
        os.remove(path)
    if CONF_DF == 'to_csv':
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        df.columns = df.columns.astype(str)
        df.to_feather(path)

def save_test_df_to_dir(df, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    path = _data_path_in_dir(base_dir, "test_data.csv" if CONF_DF == 'to_csv' else "test_data.feather")
    if os.path.exists(path):
        os.remove(path)
    if CONF_DF == 'to_csv':
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        df.columns = df.columns.astype(str)
        df.to_feather(path)

def load_train_df_from_dir(base_dir):
    path = _data_path_in_dir(base_dir, "train_data.csv" if CONF_DF == 'to_csv' else "train_data.feather")
    if CONF_DF == 'to_csv':
        return pd.read_csv(path, encoding="utf-8")
    return pd.read_feather(path)

def load_test_df_from_dir(base_dir):
    path = _data_path_in_dir(base_dir, "test_data.csv" if CONF_DF == 'to_csv' else "test_data.feather")
    if CONF_DF == 'to_csv':
        return pd.read_csv(path, encoding="utf-8")
    return pd.read_feather(path)

def get_data_config_path_in_dir(base_dir):
    return _data_path_in_dir(base_dir, "data_config_meta.json")

def load_pre_params_from_dir(base_dir) -> BaseDefine:
    """Load interval settings from data_config_meta.json under base_dir (no global paths; multiprocessing-friendly)."""
    config_path = get_data_config_path_in_dir(base_dir)
    if not os.path.exists(config_path):
        raise RuntimeError(f"❌ Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        para = BaseDefine(**meta)
    return para

def attach_attr(df, feature_group_list, feature_conf_list = [], para = BaseDefine):
    # 1. Basic preprocessing
    # df.drop('ignore', axis=1, inplace=True)
    # --- 2. Indicator computation (generate raw, unscaled feature columns) ---
    # df = add_relative_features(df)
    kline_interval_ms = get_interval_ms(para.interval)
    return FeatureFactory(kline_interval_ms,feature_group_list, feature_conf_list).generate(df)

def attach_fthl_stock_daily(df, para=BaseDefine(), label_col='label'):
    # 1. Compute asymmetric dynamic thresholds
    df = calculate_thresholds(df, para)

    n = len(df)
    df['open_time_sn'] = np.arange(len(df), dtype=np.int64)
    idx = np.arange(n)
    target_indices = idx + int(para.predict_num)

    final_valid_mask = target_indices < n
    safe_idx = np.where(final_valid_mask, target_indices, 0)

    exit_index_col = f"{label_col}_exit_index"
    exit_price_col = f"{label_col}_exit_price"

    # 2. Store exit metadata
    df[exit_index_col] = np.where(
        final_valid_mask,
        target_indices,
        -1,
    ).astype(np.int64)

    # 3. Compute forward close return
    future_close = np.where(
        final_valid_mask,
        df["close"].values[safe_idx],
        np.nan,
    )

    df[exit_price_col] = future_close

    pct_final = np.log(future_close / df["close"])

    # 4. Compute path-dependent high/low within future trading-bar window
    high_mtx = np.column_stack(
        [
            df["high"].shift(-i).values
            for i in range(1, int(para.predict_num) + 1)
        ]
    )

    low_mtx = np.column_stack(
        [
            df["low"].shift(-i).values
            for i in range(1, int(para.predict_num) + 1)
        ]
    )

    # For stock daily, valid horizon is exactly predict_num trading bars.
    # Invalid tail rows are masked by final_valid_mask.
    future_high_max = np.maximum.accumulate(high_mtx, axis=1)[
        np.arange(n),
        int(para.predict_num) - 1,
    ]

    future_low_min = np.minimum.accumulate(low_mtx, axis=1)[
        np.arange(n),
        int(para.predict_num) - 1,
    ]

    max_drawdown = (future_low_min - df["close"]) / df["close"]
    max_runup = (future_high_max - df["close"]) / df["close"]

    # 5. Apply asymmetric label logic
    cond_long = (
        final_valid_mask
        & (pct_final > df["threshold_long"])
        & (max_drawdown > -df["stop_threshold_long"])
    )

    cond_short = (
        final_valid_mask
        & (pct_final < -df["threshold_short"])
        & (max_runup < df["stop_threshold_short"])
    )

    conditions = [
        ~final_valid_mask,
        cond_short,
        cond_long,
    ]

    choices = [
        Signal.INVALID,
        Signal.NEGATIVE,
        Signal.POSITIVE,
    ]

    df[label_col] = np.select(
        conditions,
        choices,
        default=Signal.NEUTRAL,
    ).astype(int)

    return df

    # 1. 兼容时间戳：将纳秒转为通用的毫秒
import numpy as np

def attach_fthl_stock_minutely(df, para=BaseDefine(), label_col='label'):
    if 'window_start' in df.columns and 'open_time_ms_utc' not in df.columns:
        df['open_time_ms_utc'] = df['window_start'] // 1000000

    df = df.sort_values("open_time_ms_utc").reset_index(drop=True)

    # 2. 计算动态波动率阈值 (内部已剔除跳空)
    df = calculate_us_stock_mins_thresholds(df, para)

    time_col = 'open_time_ms_utc'
    time_values = df[time_col].values
    interval_ms = get_interval_ms(para.interval)
    n = len(df)
    
    # ---------------------------------------------------------
    # 修改点 1：将 open_time_sn 改为严格连续的自增标注
    # 供下游特征处理模块校验输入序列（Lookback Window）的连续性
    # ---------------------------------------------------------
    df['open_time_sn'] = np.arange(n, dtype=np.int64)

    # ---------------------------------------------------------
    # 修改点 2：严格校验预测窗口，不允许隔夜、不允许中间断层
    # ---------------------------------------------------------
    idx = np.arange(n)
    # 强制让目标索引就是往后数 predict_num 根 K 线
    target_indices = idx + int(para.predict_num)
    
    in_bounds = target_indices < n
    safe_idx = np.where(in_bounds, target_indices, 0)
    
    # 预期中“完美的”未来物理时间
    target_times = time_values + (para.predict_num * interval_ms)
    
    # 灵魂判定：只有当向后数 predict_num 根 K 线的时间，
    # 刚好等同于理想的 target_times 时，才算合法。
    # 一旦中间发生隔夜跳空或缺失，物理时间一定会严重大于 target_times，被无情过滤为 False。
    final_valid_mask = in_bounds & (time_values[safe_idx] == target_times)

    exit_index_col = f"{label_col}_exit_index"
    df[exit_index_col] = np.where(final_valid_mask, target_indices, -1).astype(np.int64)

    exit_price_col = f"{label_col}_exit_price"
    future_close = np.where(final_valid_mask, df['close'].values[safe_idx], np.nan)
    df[exit_price_col] = future_close
    pct_final = np.log(future_close / df['close'])

    # 计算路径极值：
    # 因为现在的 target_indices 严格等于当前 index + predict_num，
    # 所以直接取所有 predict_num 列的最大最小值即可，无需再做复杂的 clip 截断
    high_mtx = np.column_stack([df['high'].shift(-i).values for i in range(1, para.predict_num + 1)])
    low_mtx = np.column_stack([df['low'].shift(-i).values for i in range(1, para.predict_num + 1)])
    
    future_high_max = np.maximum.accumulate(high_mtx, axis=1)[:, para.predict_num - 1]
    future_low_min = np.minimum.accumulate(low_mtx, axis=1)[:, para.predict_num - 1]

    max_drawdown = (future_low_min - df['close']) / df['close']
    max_runup = (future_high_max - df['close']) / df['close']

    # 4. 非对称标签逻辑
    cond_long = final_valid_mask & (pct_final > df['threshold_long']) & (max_drawdown > -df['stop_threshold_long'])
    cond_short = final_valid_mask & (pct_final < -df['threshold_short']) & (max_runup < df['stop_threshold_short'])

    conditions = [~final_valid_mask, cond_short, cond_long]
    choices = [Signal.INVALID, Signal.NEGATIVE, Signal.POSITIVE]
    df[label_col] = np.select(conditions, choices, default=Signal.NEUTRAL).astype(int)
    
    return df

def attach_label(df, para = BaseDefine(), label_col = 'label'):
    if para.label_type == 'FTHL':
        return attach_fthl_label(df, para, label_col)
    elif para.label_type == 'TBM':
        return attach_tbm_label(df, para, label_col)
    else:
        raise ValueError

# fixed-time horizon labeling
def attach_fthl_label(df, para = BaseDefine(), label_col = 'label'):
    time_col = 'open_time_ms_utc'
    interval_ms = get_interval_ms(para.interval)
    if para.market_category == "Stock":
        if 'd' in para.interval:
            return attach_fthl_stock_daily(df, para, label_col)
        elif 'm' in para.interval:
            return attach_fthl_stock_minutely(df, para, label_col)
        df['open_time_sn'] = df[time_col]// interval_ms
    elif para.symbol == "XAUUSD":
        #Weekend gaps were not separately removed, since they account for a very small proportion of the observations. However, this may introduce a small amount of calendar-time inconsistency for prediction windows crossing weekends.
        df['open_time_sn'] = np.arange(len(df), dtype=np.int64)
    else:
        df['open_time_sn'] = df[time_col]// interval_ms
    """
    Path-dependent asymmetric labeling logic.
    """
    time_values = df[time_col].values
    
    # 1. Compute asymmetric dynamic thresholds
    df = calculate_thresholds(df, para)

    # 2. Physical time anchoring (unchanged)
    df['open_time_sn'] = df[time_col]// interval_ms
    target_times = time_values + (para.predict_num * interval_ms)
    target_indices = np.searchsorted(time_values, target_times, side='left')
    in_bounds = target_indices < len(df)
    safe_idx = np.where(in_bounds, target_indices, 0)
    final_valid_mask = in_bounds & (time_values[safe_idx] == target_times)

    exit_index_col = f"{label_col}_exit_index"
    # This is safer than using i + predict_num because K-lines may be missing.
    df[exit_index_col] = np.where(final_valid_mask, target_indices, -1).astype(np.int64)

    exit_price_col = f"{label_col}_exit_price"

    # 3. Compute forward return and extreme moves (unchanged)
    future_close = np.where(final_valid_mask, df['close'].values[safe_idx], np.nan)

    df[exit_price_col] = future_close
    pct_final = np.log(future_close / df['close'])

    high_mtx = np.column_stack([df['high'].shift(-i).values for i in range(1, para.predict_num + 1)])
    low_mtx = np.column_stack([df['low'].shift(-i).values for i in range(1, para.predict_num + 1)])
    
    steps = (target_indices - np.arange(len(df))).clip(1, para.predict_num)
    future_high_max = np.maximum.accumulate(high_mtx, axis=1)[np.arange(len(df)), steps - 1]
    future_low_min = np.minimum.accumulate(low_mtx, axis=1)[np.arange(len(df)), steps - 1]

    max_drawdown = (future_low_min - df['close']) / df['close']
    max_runup = (future_high_max - df['close']) / df['close']

    # 4. Apply asymmetric logic
    # Long: use long-side thresholds
    cond_long = final_valid_mask & \
                (pct_final > df['threshold_long']) & \
                (max_drawdown > -df['stop_threshold_long'])
                
    # Short: use short-side thresholds
    cond_short = final_valid_mask & \
                 (pct_final < -df['threshold_short']) & \
                 (max_runup < df['stop_threshold_short'])

    # 5. Build labels
    conditions = [~final_valid_mask, cond_short, cond_long]
    choices = [Signal.INVALID, Signal.NEGATIVE, Signal.POSITIVE ]
    df[label_col] = np.select(conditions, choices, default=Signal.NEUTRAL).astype(int)
    
    return df

# Advances in Financial Machine Learning by Dr. Marcos López de Prado (2018) introduced the triple barrier method for labeling financial data, which is a more sophisticated approach than simple return-based labeling. The method considers both profit-taking and stop-loss barriers, as well as a time limit, to determine the label of each sample. Below is an implementation of the triple barrier method in Python, using Numba for performance optimization.
def calculate_thresholds(df, para=BaseDefine, **kwargs):
    """
    Compute dynamic volatility thresholds using
    EWMA standard deviation of candle-to-candle simple returns.
    """

    assert 'close' in df.columns, "Missing column: close"

    df = df.copy()
    df['ret'] = df['close'] / df['close'].shift(1) - 1
    ewma_vol = df['ret'].ewm(span=para.vol_ewma_span, adjust=False).std()
    # expected_vol = ewma_vol * np.sqrt(para.predict_num)
    expected_vol = ewma_vol
    df['expected_vol'] = expected_vol

    # ===== 4️⃣ Asymmetric thresholds =====
    df['threshold_long'] = expected_vol * para.vol_multiplier_long
    df['threshold_short'] = expected_vol * para.vol_multiplier_short

    if para.stop_multiplier_rate_long is not None:
        df['stop_threshold_long'] = df['threshold_long'] * para.stop_multiplier_rate_long
    else:
        df['stop_threshold_long'] = np.inf

    if para.stop_multiplier_rate_short is not None:
        df['stop_threshold_short'] = df['threshold_short'] * para.stop_multiplier_rate_short
    else:
        df['stop_threshold_short'] = np.inf

    return df

def calculate_us_stock_mins_thresholds(df, para=BaseDefine):
    """
    计算动态波动率阈值。
    新增逻辑：剔除美股隔夜跳空缺口对 EWMA 波动率的污染。
    """
    assert 'close' in df.columns, "Missing column: close"
    df = df.copy()
    
    # 1. 计算原始的 K线到K线 收益率
    df['ret'] = df['close'] / df['close'].shift(1) - 1

    # 2. 识别“每日首根 K 线”以剔除隔夜跳空
    if 'datetime_utc' in df.columns:
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
        # 确保时区正确，转换为美东时间来判断自然日
        if df['datetime_utc'].dt.tz is None:
            dt_est = pd.to_datetime(df['datetime_utc'], utc=True).dt.tz_convert('US/Eastern')
        else:
            dt_est = df['datetime_utc'].dt.tz_convert('US/Eastern')
        
        # 提取日期 (YYYY-MM-DD)
        trade_dates = dt_est.dt.date
        
        # 如果当前行的日期和上一行不一样，说明这是新的一天的第一根 K 线
        is_new_day = trade_dates != trade_dates.shift(1)
        
        # 将跳空收益率设为 NaN，这样 ewm.std() 会自动忽略它，不会污染波动率均值
        df.loc[is_new_day, 'ret'] = np.nan
        
    # 3. 计算 EWMA 波动率 (忽略 NaN)
    ewma_vol = df['ret'].ewm(span=para.vol_ewma_span, adjust=False, ignore_na=True).std()
    
    expected_vol = ewma_vol
    df['expected_vol'] = expected_vol

    # ===== 4️⃣ 非对称阈值 =====
    df['threshold_long'] = expected_vol * para.vol_multiplier_long
    df['threshold_short'] = expected_vol * para.vol_multiplier_short

    if getattr(para, 'stop_multiplier_rate_long', None) is not None:
        df['stop_threshold_long'] = df['threshold_long'] * para.stop_multiplier_rate_long
    else:
        df['stop_threshold_long'] = np.inf

    if getattr(para, 'stop_multiplier_rate_short', None) is not None:
        df['stop_threshold_short'] = df['threshold_short'] * para.stop_multiplier_rate_short
    else:
        df['stop_threshold_short'] = np.inf

    return df

@njit(cache=True)
def fast_triple_barrier_kernel(close, high, low, thresholds, window):
    n = len(close)
    labels = np.ones(n, dtype=np.int32)         # 默认中性 (1)
    reach_times = np.full(n, window, dtype=np.int32) # 默认到期窗口长度

    l_tp_p = thresholds[:, 0]
    l_sl_p = thresholds[:, 1]
    s_tp_p = thresholds[:, 2]
    s_sl_p = thresholds[:, 3]

    for i in range(n - window):
        p0 = close[i]
        
        # 独立的价格屏障
        l_tp = p0 * (1 + l_tp_p[i])
        l_sl = p0 * (1 - l_sl_p[i])
        s_tp = p0 * (1 - s_tp_p[i])
        s_sl = p0 * (1 + s_sl_p[i])
        
        # 用来记录该样本在窗口内达成 TP 或 SL 的最早步数
        first_l_tp = window + 1
        first_s_tp = window + 1
        first_l_sl = window + 1
        first_s_sl = window + 1
        
        l_active = True
        s_active = True

        for j in range(1, window + 1):
            curr_idx = i + j
            h, l = high[curr_idx], low[curr_idx]
            
            # --- 多头路径判定 ---
            if l_active:
                hit_l_tp = (h >= l_tp)
                hit_l_sl = (l <= l_sl)
                
                if hit_l_tp and hit_l_sl:
                    # 极端波动：单根K线同时触碰止盈与止损
                    # 秉持悲观原则，假设先被止损
                    first_l_sl = j
                    l_active = False 
                elif hit_l_sl:
                    first_l_sl = j
                    l_active = False 
                elif hit_l_tp:
                    first_l_tp = j
                    l_active = False

            # --- 空头路径判定 ---
            if s_active:
                hit_s_tp = (l <= s_tp)
                hit_s_sl = (h >= s_sl)
                
                if hit_s_tp and hit_s_sl:
                    # 极端波动：单根K线同时触碰止盈与止损
                    # 秉持悲观原则，假设先被止损
                    first_s_sl = j
                    s_active = False
                elif hit_s_sl:
                    first_s_sl = j
                    s_active = False
                elif hit_s_tp:
                    first_s_tp = j
                    s_active = False

            # 如果多空都已经有了明确结果（无论触碰了TP还是SL），提前退出循环
            if not l_active and not s_active:
                break

        # --- 最终决策逻辑 ---
        # 规则 1：多头获胜条件 —— 多头触碰过 TP，且步数严格领先于空头触发 TP 的步数
        if first_l_tp <= window and first_l_tp < first_s_tp:
            labels[i] = 2 # Signal.POSITIVE
            reach_times[i] = first_l_tp
            
        # 规则 2：空头获胜条件 —— 空头触碰过 TP，且步数严格领先于多头触发 TP 的步数
        elif first_s_tp <= window and first_s_tp < first_l_tp:
            labels[i] = 0 # Signal.NEGATIVE
            reach_times[i] = first_s_tp
            
        # 规则 3：双输、双触或均未触发 —— 判定为中性
        else:
            labels[i] = 1 # Signal.NEUTRAL
            # 如果是中性标签，这里记录下最早发生止损的步数，方便回测引擎做非重叠仓位的时间跨度截断
            min_sl = min(first_l_sl, first_s_sl)
            if min_sl <= window:
                reach_times[i] = min_sl
            else:
                reach_times[i] = window
                
    return labels, reach_times

def attach_tbm_label(df, para=BaseDefine(), label_col = 'label'):
    interval_ms = get_interval_ms(para.interval)
    if para.market_category == "Stock":
        df['open_time_sn'] = np.arange(len(df), dtype=np.int64)
        if 'd' in para.interval:
            df = calculate_thresholds(df, para)
        elif 'm' in para.interval:
            df = calculate_us_stock_mins_thresholds(df, para)
    elif para.market_category == "Cryptocurrency":
        df['open_time_sn'] = df['open_time_ms_utc']// interval_ms
        df = calculate_thresholds(df, para)
    elif para.symbol == "XAUUSD":
        df['open_time_sn'] = np.arange(len(df), dtype=np.int64)
        df = calculate_thresholds(df, para)
    # 2. 准备底层数据
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    
    thresholds = np.column_stack([
        df['threshold_long'].values,
        df['stop_threshold_long'].values,
        df['threshold_short'].values,
        df['stop_threshold_short'].values
    ]).astype(np.float64)
    
    window = int(para.predict_num)
    
    labels, reach_times = fast_triple_barrier_kernel(
        close, high, low, thresholds, window
    )
    
    df[label_col] = labels
    df['reach_time'] = reach_times
    
    sn_values = df['open_time_sn'].values

    target_sn = sn_values + window
    target_indices = np.searchsorted(sn_values, target_sn, side='left')

    in_bounds = target_indices < len(df)

    time_match = np.zeros(len(df), dtype=np.bool_)
    valid_idx = np.where(in_bounds)[0]

    time_match[valid_idx] = (
        sn_values[target_indices[valid_idx]] == target_sn[valid_idx]
    )
    
    df.loc[~time_match, label_col] = -1       # Signal.INVALID
    df.loc[~time_match, 'reach_time'] = -1  # 无效到达时间

    df[f"{label_col}_threshold_long"] = df["threshold_long"]
    df[f"{label_col}_threshold_short"] = df["threshold_short"]
    df[f"{label_col}_stop_threshold_long"] = df["stop_threshold_long"]
    df[f"{label_col}_stop_threshold_short"] = df["stop_threshold_short"]

    return df

def print_label_performance_stats(df, para=BaseDefine):
    """
    打印标签分布及到达时间的深度统计信息
    """
    print("\n" + "="*20 + " 📊 Triple Barrier Statistics " + "="*20)
    
    # 1. 基础信息
    total_len = len(df)
    valid_df = df[df['label'] != -1].copy() # 排除 INVALID (-1)
    predict_num = para.predict_num
    
    print(f"Total Samples: {total_len}")
    print(f"Valid Samples: {len(valid_df)} ({(len(valid_df)/total_len)*100:.2f}%)")
    print(f"Max Window (predict_num): {predict_num}")
    print("-" * 50)

    # 2. 标签分布统计
    label_counts = valid_df['label'].value_counts().sort_index()
    label_map = {0: "NEGATIVE (Short Win)", 1: "NEUTRAL (Time-out/SL)", 2: "POSITIVE (Long Win)"}
    
    print(f"{'Label Type':<25} | {'Count':<10} | {'Percentage':<10}")
    for lbl, count in label_counts.items():
        name = label_map.get(lbl, "Unknown")
        pct = (count / len(valid_df)) * 100
        print(f"{name:<25} | {count:<10} | {pct:>8.2f}%")
    
    print("-" * 50)

    # 3. Reach Time 统计 (针对非中性标签)
    print("⏱️ Reach Time Descriptive Statistics (Steps):")
    
    # 分组计算 reach_time 的描述性统计
    stats = valid_df.groupby('label')['reach_time'].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9]
    )
    # 重命名索引方便阅读
    stats.index = stats.index.map(label_map)
    print(stats[['count', 'mean', 'min', '50%', '90%', 'max']])

    # 4. 效率分析：快速触发 vs 慢速触发
    print("\n🚀 Efficiency Analysis (Speed of Signal):")
    for lbl in [0, 2]:
        sub = valid_df[valid_df['label'] == lbl]
        if len(sub) > 0:
            name = label_map[lbl]
            # 定义“快速触发”为在窗口前 25% 的时间内就达标
            fast_threshold = predict_num * 0.25
            fast_hits = len(sub[sub['reach_time'] <= fast_threshold])
            fast_pct = (fast_hits / len(sub)) * 100
            
            # 定义“压哨触发”为在窗口最后 10% 的时间内才达标
            slow_threshold = predict_num * 0.9
            slow_hits = len(sub[sub['reach_time'] >= slow_threshold])
            
            print(f"[{name}]")
            print(f"  - Fast Hits (<= {fast_threshold:.0f} steps): {fast_hits} ({fast_pct:.2f}%)")
            print(f"  - Slow Hits (>= {slow_threshold:.0f} steps): {slow_hits} ({(slow_hits/len(sub))*100:.2f}%)")
            print(f"  - Median Reach Time: {sub['reach_time'].median():.0f} steps")

    print("="*60 + "\n")


def clean_data_quality_auto(df: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Starting automated data quality scan...")
    initial_count = len(df)
    na_rows = df.isna().any(axis=1).sum()
    if na_rows > 0:
        logger.warning(f"Detected {na_rows} rows containing NaN values; dropping them.")

    price_cols = ['open','close','high','low']
    # price_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    zero_mask = (df[price_cols] == 0).any(axis=1)
    zero_rows = zero_mask.sum()
    
    if zero_rows > 0:
        zero_stats = (df[price_cols] == 0).sum()
        logger.warning(f"Detected {zero_rows} rows containing zero values. Distribution:\n{zero_stats[zero_stats > 0]}")

    condition = df.isna().any(axis=1) | zero_mask
    df_cleaned = df[~condition].copy()
    df_cleaned.reset_index(drop=True, inplace=True)

    final_count = len(df_cleaned)
    dropped_count = initial_count - final_count

    if dropped_count > 0:
        logger.info(f"✅ Cleaning done: {initial_count} rows -> {final_count} rows (dropped {dropped_count})")
    else:
        logger.info("✅ Scan complete: no NaN or zero values found.")

    return df_cleaned

def float_range(start, end, step):
    values = []
    v = start
    eps = step / 10
    while v <= end + eps:
        values.append(round(v, 10))
        v += step
    return values

@lru_cache(maxsize=1)
def load_interval_ms(config_path = data_config_path):
    if not os.path.exists(config_path):
        raise RuntimeError(f"❌ Config file not found: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        interval_ms = meta.get("interval_ms")
        if interval_ms is None:
            raise RuntimeError("⚠️ Missing 'interval_ms' field in config file!")
        return interval_ms
    except Exception as e:
        raise RuntimeError(f"💥 Unexpected error while reading JSON: {e}")

def setup_session_logger(sub_folder: str = None, log_file_path=None, symbol: str = BaseDefine.symbol, console_level: int = logging.INFO, file_level: int = logging.INFO):
    if log_file_path ==None:
        assert sub_folder!=None
        log_dir = os.path.join(OUTPUT_DIR,'log', sub_folder)
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sym_str = f"_{symbol}" if symbol else ""
        log_filename = f"session{sym_str}_{timestamp}.log"
        log_file_path = os.path.join(log_dir, log_filename)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) 
    if root_logger.handlers:
        root_logger.handlers = []
    log_format_console = "%(log_color)s%(asctime)s-%(name)s-%(levelname)s- %(message)s"
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    color_formatter = colorlog.ColoredFormatter(
        log_format_console,
        datefmt="%H:%M:%S",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'RECORD':   'blue',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red,bg_yellow',
        }
    )
    ch.setFormatter(color_formatter)
    root_logger.addHandler(ch)
    fh = logging.FileHandler(log_file_path, encoding='utf-8')
    fh.setLevel(file_level) 
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(file_formatter)
    root_logger.addHandler(fh)
    root_logger.info(f"Session Logger Initialized. Log file: {log_file_path}")
    return root_logger, log_file_path

def get_interval_from_filename(path: str) -> str:
    """
    Extract interval string from a file path (e.g. ETHUSDT_3m.csv -> 3m).
    """
    filename = os.path.basename(path)
    # Match formats like 1s, 15s, 1m, 3m... 1M
    match = re.search(r'_(\d+[smhdwM])\.csv', filename)
    if match:
        return match.group(1)
    return "unknown"

def get_interval_ms(interval_str: str) -> int:
    """
    Convert an interval string to milliseconds.
    Supported: 1s, 15s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """
    # Base units in milliseconds
    units = {
        's': 1000,
        'm': 60 * 1000,
        'h': 60 * 60 * 1000,
        'd': 24 * 60 * 60 * 1000,
        'w': 7 * 24 * 60 * 60 * 1000,
        'M': 30 * 24 * 60 * 60 * 1000  # Approximate month as 30 days
    }
    
    # Split number and unit via regex
    match = re.match(r'(\d+)([smhdwM])', interval_str)
    if not match:
        return 0
    
    value, unit = match.groups()
    return int(value) * units[unit]

def get_git_info(logger):
    repo = git.Repo(PROJECT_DIR)
    sha = repo.head.object.hexsha
    short_sha = repo.git.rev_parse(sha, short=8)
    
    logger.info(f"Full SHA: {sha}")
    logger.info(f"Short SHA: {short_sha}")
    logger.info(f"Commit Message: {repo.head.object.message.strip()}")
    return short_sha

def save_params(path, *, strategy, common, train):
    data = {
        "strategy": asdict(strategy),
        "common": asdict(common),
        "train": asdict(train),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def build_dataclass(cls, data: dict):
    """
    Build a dataclass from a dict (supports nested dataclasses).
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue

        val = data[f.name]

        # Nested dataclass
        if is_dataclass(f.type) and isinstance(val, dict):
            kwargs[f.name] = build_dataclass(f.type, val)
        else:
            kwargs[f.name] = val

    return cls(**kwargs)

def load_parameters(path, cls):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return build_dataclass(cls, data["strategy"])

def load_common_define(path, cls):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return build_dataclass(cls, data["common"])

def load_train_config(path, cls):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return build_dataclass(cls, data["train"])

def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())   # Optional but recommended