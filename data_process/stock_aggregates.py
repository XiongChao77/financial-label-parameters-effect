import pandas as pd
from pathlib import Path

stock_input = Path("QuantData/Stock/massive/spot/QQQ_1m.csv")

df = pd.read_csv(stock_input)

df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
df = df.sort_values("datetime_utc").reset_index(drop=True)


def aggregate_kbars(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    data = data.copy()

    data["datetime_utc"] = pd.to_datetime(data["datetime_utc"], utc=True)
    data = data.sort_values("datetime_utc").reset_index(drop=True)

    # 用美东时间过滤美股正常交易时间
    data["datetime_et"] = data["datetime_utc"].dt.tz_convert("America/New_York")

    data = (
        data.set_index("datetime_et")
            .between_time("09:30", "16:00", inclusive="left")
            .reset_index()
    )

    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "transactions": "sum",
    }

    resampled = (
        data.set_index("datetime_et")
            .groupby("ticker")
            .resample(timeframe, label="left", closed="left")
            .agg(agg_dict)
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
    )

    # 聚合后的K线起始时间：美东 -> UTC
    resampled["datetime_utc"] = resampled["datetime_et"].dt.tz_convert("UTC")

    # 保留/重新生成全部时间字段
    resampled["window_start"] = resampled["datetime_utc"].astype("int64")
    resampled["open_time_ms_utc"] = resampled["window_start"] // 1_000_000
    resampled["open_time_date_utc"] = (
        resampled["datetime_utc"]
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    resampled["volume"] = resampled["volume"].astype("int64")
    resampled["transactions"] = resampled["transactions"].astype("int64")
    resampled["window_start"] = resampled["window_start"].astype("int64")
    resampled["open_time_ms_utc"] = resampled["open_time_ms_utc"].astype("int64")

    columns_order = [
        "ticker",
        "volume",
        "open",
        "close",
        "high",
        "low",
        "window_start",
        "transactions",
        "datetime_utc",
        "open_time_ms_utc",
        "open_time_date_utc",
    ]

    return (
        resampled[columns_order]
        .sort_values("datetime_utc")
        .reset_index(drop=True)
    )

df_5m = aggregate_kbars(df, "5min")
df_15m = aggregate_kbars(df, "15min")

df_5m.to_csv("QuantData/Stock/massive/spot/QQQ_5m.csv", index=False)
df_15m.to_csv("QuantData/Stock/massive/spot/QQQ_15m.csv", index=False)

print("=== 5m ===")
print(df_5m.head())

print("=== 15m ===")
print(df_15m.head())