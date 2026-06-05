import os
import pandas as pd
import numpy as np


def adjust_stock_ohlc_csv(
    input_csv_path: str,
    output_csv_path: str | None = None,
):
    """
    Read Yahoo Finance style stock CSV and convert raw OHLC
    into adjusted OHLC using:

        adjust_factor = Adj Close / Close

    Input columns:
        Date, Adj Close, Close, High, Low, Open, Volume

    Output columns (Lowercased):
        date, open, high, low, close, volume

    Notes:
        - Output OHLC are all adjusted to the same scale.
        - Volume is kept unchanged.
        - Rows with invalid Close are removed.
    """

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------
    df = pd.read_csv(input_csv_path)

    required_cols = [
        "Date",
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ---------------------------------------------------------
    # Clean
    # ---------------------------------------------------------
    df = df.copy()

    numeric_cols = [
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove invalid rows
    df = df[
        np.isfinite(df["Close"])
        & np.isfinite(df["Adj Close"])
        & (df["Close"] > 0)
    ].copy()

    # ---------------------------------------------------------
    # Compute adjustment factor
    # ---------------------------------------------------------
    adj_factor = df["Adj Close"] / df["Close"]

    # ---------------------------------------------------------
    # Adjust OHLC
    # ---------------------------------------------------------
    df["Open"] = df["Open"] * adj_factor
    df["High"] = df["High"] * adj_factor
    df["Low"] = df["Low"] * adj_factor
    df["Close"] = df["Adj Close"]

    # ---------------------------------------------------------
    # Keep only adjusted OHLCV
    # ---------------------------------------------------------
    out_df = df[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ].copy()

    # --- 新增步骤：将列名转换为小写 ---
    out_df.columns = out_df.columns.str.lower()

    # ---------------------------------------------------------
    # Sort by date
    # ---------------------------------------------------------
    # 注意：这里因为列名已经变了，所以要用小写的 'date'
    out_df["date"] = pd.to_datetime(out_df["date"])
    out_df = out_df.sort_values("date").reset_index(drop=True)

    # ---------------------------------------------------------
    # Output path
    # ---------------------------------------------------------
    if output_csv_path is None:
        base, ext = os.path.splitext(input_csv_path)
        output_csv_path = f"{base}_adjusted.csv"

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    out_df.to_csv(output_csv_path, index=False)

    print(f"Saved adjusted CSV:")
    print(output_csv_path)
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":

    # Example
    input_csv = r"/home/chao/work/QuantData/Stock/yfinance/spot/spy_1d_origin.csv"

    adjust_stock_ohlc_csv(input_csv)