import time,os
from pathlib import Path
import pandas as pd

current_work_dir = os.path.dirname(__file__) 

TARGET = "QQQ"

# QQQ 以前曾经用过 QQQQ，所以这里两个都找
TARGET_CANDIDATES = ["QQQ", "QQQQ"]

period = 'day' # 'day'
DATA_DIR = Path(os.path.join(current_work_dir, '..', 'data', period))
SPLITS_PATH = Path(os.path.join(current_work_dir, '..', 'data', "massive_splits_all.csv"))

if period == 'minute':
    prefix = '1m'
elif period == 'day':
    prefix = '1d'
RAW_OUTPUT_PATH = Path(f"{TARGET.lower()}_{prefix}_raw.csv")
ADJUSTED_OUTPUT_PATH = Path(f"{TARGET.lower()}_{prefix}_split_adjusted.csv")

# Massive day aggregates 常见列名：
# ticker, volume, open, close, high, low, window_start, transactions
PRICE_COLS = ["open", "high", "low", "close"]
VOLUME_COLS = ["volume"]

# =========================
# 工具函数
# =========================

def check_missing_periods(df: pd.DataFrame, datetime_col: str = "datetime_utc") -> None:
    """
    检查 DataFrame 中的日期序列是否缺失了某些年份或月份。
    注意：这里只检查月份是否存在，不检查每个交易日是否完整。
    """
    if df.empty:
        print("警告: 数据集为空，无法检查缺失。")
        return

    min_date = df[datetime_col].min()
    max_date = df[datetime_col].max()

    print("\n" + "=" * 40)
    print(f"数据时间范围: {min_date.strftime('%Y-%m')} 至 {max_date.strftime('%Y-%m')}")

    expected_months = pd.date_range(
        start=min_date,
        end=max_date,
        freq="MS"
    ).to_period("M")

    actual_months = pd.PeriodIndex(df[datetime_col].dt.to_period("M").unique())

    missing_months = expected_months.difference(actual_months)

    if len(missing_months) == 0:
        print("✅ 检查通过：在数据范围内，没有缺失任何月份或年份！")
    else:
        print(f"❌ 警告：发现缺失！在当前时间范围内，共缺失了 {len(missing_months)} 个月份：")

        missing_dict = {}
        for m in missing_months:
            missing_dict.setdefault(m.year, []).append(m.month)

        for year, months in sorted(missing_dict.items()):
            if len(months) == 12:
                print(f"  - 缺失整年: {year} 年")
            else:
                months_str = ", ".join(f"{m:02d}月" for m in sorted(months))
                print(f"  - {year} 年缺失月份: {months_str}")

    print("=" * 40 + "\n")

def load_target_stock_data(
    data_dir: Path,
    target_candidates: list[str],
) -> pd.DataFrame:
    """
    从 Massive day flat files 中提取目标 ticker 数据。
    """
    files = sorted(data_dir.rglob("*.csv.gz"))

    if not files:
        raise FileNotFoundError(f"No .csv.gz files found in: {data_dir.resolve()}")

    target_data_list = []

    print("=" * 80)
    print("Start scanning Massive daily files")
    print(f"Data dir: {data_dir.resolve()}")
    print(f"Target candidates: {target_candidates}")
    print(f"File count: {len(files)}")
    print("=" * 80)

    start_time = time.time()

    for i, file in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Processing: {file.name}")

        df = pd.read_csv(file)

        if "ticker" not in df.columns:
            raise ValueError(f"File {file} does not contain column: ticker")

        target_data = df[df["ticker"].isin(target_candidates)].copy()

        if not target_data.empty:
            target_data_list.append(target_data)
            print(f"  found rows: {len(target_data)}")

    if not target_data_list:
        raise ValueError(f"No data found for target candidates: {target_candidates}")

    target_data_all = pd.concat(target_data_list, ignore_index=True)

    if "window_start" not in target_data_all.columns:
        raise ValueError("Target data does not contain column: window_start")

    target_data_all = target_data_all.sort_values("window_start").reset_index(drop=True)

    # Massive flat files 的 window_start 通常是 nanoseconds since epoch
    target_data_all["datetime_utc"] = pd.to_datetime(
        target_data_all["window_start"],
        unit="ns",
        utc=True,
    )

    target_data_all["date"] = target_data_all["datetime_utc"].dt.strftime("%Y-%m-%d")

    print("=" * 80)
    print("Finished scanning")
    print(f"Rows found: {len(target_data_all)}")
    print(f"Time used: {time.time() - start_time:.2f}s")
    print("=" * 80)

    return target_data_all


def load_splits_for_target(
    splits_path: Path,
    target_candidates: list[str],
) -> pd.DataFrame:
    """
    读取 split 文件，并筛选目标 ticker 的 split 事件。

    massive_splits_all.csv 格式示例：
    id,execution_date,split_from,split_to,ticker,adjustment_type,historical_adjustment_factor
    """
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path.resolve()}")

    splits = pd.read_csv(splits_path)

    required_cols = [
        "execution_date",
        "split_from",
        "split_to",
        "ticker",
    ]

    missing_cols = [c for c in required_cols if c not in splits.columns]
    if missing_cols:
        raise ValueError(f"Splits file missing columns: {missing_cols}")

    splits = splits[splits["ticker"].isin(target_candidates)].copy()

    if splits.empty:
        print("=" * 80)
        print("No split events found for target.")
        print(f"Target candidates: {target_candidates}")
        print("=" * 80)
        return splits

    splits["execution_date"] = pd.to_datetime(
        splits["execution_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    splits["split_from"] = pd.to_numeric(splits["split_from"], errors="coerce")
    splits["split_to"] = pd.to_numeric(splits["split_to"], errors="coerce")

    splits = splits.dropna(subset=["execution_date", "split_from", "split_to"])

    # 单次 split 的价格调整因子：
    # 例如 2-for-1 split:
    # split_from = 1, split_to = 2
    # 历史价格乘 1/2，历史成交量乘 2
    splits["price_factor"] = splits["split_from"] / splits["split_to"]
    splits["volume_factor"] = splits["split_to"] / splits["split_from"]

    splits = splits.sort_values("execution_date").reset_index(drop=True)

    print("=" * 80)
    print("Split events for target:")
    print(splits)
    print("=" * 80)

    return splits


def apply_split_adjustment(
    df: pd.DataFrame,
    splits: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    对 OHLCV 做 split 后复权。

    原始列保留：
    open, high, low, close, volume

    新增列：
    split_adj_factor_price
    split_adj_factor_volume
    adj_open
    adj_high
    adj_low
    adj_close
    adj_volume

    复权规则：
    对 split execution_date 之前的历史数据进行调整。
    即：
    bar_date < execution_date 的行才应用该次 split。
    """

    adjusted = df.copy()

    if date_col not in adjusted.columns:
        raise ValueError(f"Data does not contain date column: {date_col}")

    adjusted[date_col] = pd.to_datetime(adjusted[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    adjusted["split_adj_factor_price"] = 1.0
    adjusted["split_adj_factor_volume"] = 1.0

    if splits.empty:
        print("No splits to apply. Adjusted prices equal raw prices.")
    else:
        for _, split in splits.iterrows():
            execution_date = split["execution_date"]
            price_factor = float(split["price_factor"])
            volume_factor = float(split["volume_factor"])
            split_ticker = split["ticker"]

            mask = adjusted[date_col] < execution_date

            affected_rows = int(mask.sum())

            print(
                f"Applying split: ticker={split_ticker}, "
                f"execution_date={execution_date}, "
                f"split_from={split['split_from']}, "
                f"split_to={split['split_to']}, "
                f"price_factor={price_factor:.10f}, "
                f"volume_factor={volume_factor:.10f}, "
                f"affected_rows={affected_rows}"
            )

            adjusted.loc[mask, "split_adj_factor_price"] *= price_factor
            adjusted.loc[mask, "split_adj_factor_volume"] *= volume_factor

    for col in PRICE_COLS:
        if col in adjusted.columns:
            adjusted[f"adj_{col}"] = adjusted[col] * adjusted["split_adj_factor_price"]
        else:
            print(f"Warning: price column not found: {col}")

    for col in VOLUME_COLS:
        if col in adjusted.columns:
            adjusted[f"adj_{col}"] = adjusted[col] * adjusted["split_adj_factor_volume"]
        else:
            print(f"Warning: volume column not found: {col}")

    return adjusted


def print_summary(df_raw: pd.DataFrame, df_adj: pd.DataFrame) -> None:
    """
    打印结果摘要。
    """
    print("\n" + "=" * 80)
    print("Raw data summary")
    print("=" * 80)
    print("Rows:", len(df_raw))
    print("Tickers:", sorted(df_raw["ticker"].unique()))
    print("Date range:", df_raw["date"].min(), "to", df_raw["date"].max())
    print(df_raw.head())
    print(df_raw.tail())

    print("\n" + "=" * 80)
    print("Adjusted data summary")
    print("=" * 80)
    print("Rows:", len(df_adj))
    print("Tickers:", sorted(df_adj["ticker"].unique()))
    print("Date range:", df_adj["date"].min(), "to", df_adj["date"].max())

    display_cols = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "split_adj_factor_price",
        "split_adj_factor_volume",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "adj_volume",
    ]

    display_cols = [c for c in display_cols if c in df_adj.columns]

    print(df_adj[display_cols].head())
    print(df_adj[display_cols].tail())
    print("=" * 80 + "\n")


# =========================
# 主程序
# =========================

if __name__ == "__main__":
    t0 = time.time()

    target_data_all = load_target_stock_data(
        data_dir=DATA_DIR,
        target_candidates=TARGET_CANDIDATES,
    )

    check_missing_periods(target_data_all)

    target_data_all.to_csv(RAW_OUTPUT_PATH, index=False)
    print("Saved raw data to:", RAW_OUTPUT_PATH.resolve())

    splits_target = load_splits_for_target(
        splits_path=SPLITS_PATH,
        target_candidates=TARGET_CANDIDATES,
    )

    target_data_adjusted = apply_split_adjustment(
        df=target_data_all,
        splits=splits_target,
        date_col="date",
    )

    target_data_adjusted = target_data_adjusted.sort_values("window_start").reset_index(drop=True)

    target_data_adjusted.to_csv(ADJUSTED_OUTPUT_PATH, index=False)
    print("Saved split-adjusted data to:", ADJUSTED_OUTPUT_PATH.resolve())

    print_summary(target_data_all, target_data_adjusted)

    print(f"Total time used: {time.time() - t0:.2f}s")