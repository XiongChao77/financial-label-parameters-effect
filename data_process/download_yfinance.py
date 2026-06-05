import yfinance as yf
import pandas as pd
import os

def download_and_save_separately(tickers, period="max"):
    """
    下载资产数据并分别保存为独立的 CSV
    """
    print(f"正在下载数据: {tickers} ...")
    
    # 获取数据
    # auto_adjust=False 确保 'Adj Close' 存在
    df = yf.download(
        tickers, 
        period=period, 
        interval="1d", 
        auto_adjust=False, 
        group_by='column'
    )

    if df.empty:
        print("下载失败，请检查网络。")
        return

    # 循环保存每个 Ticker
    for ticker in tickers:
        print(f"正在处理 {ticker}...")
        
        # 使用 .xs (cross-section) 从 MultiIndex 中提取特定 Ticker 的所有列
        # level=1 对应的是 Ticker 这一层索引
        ticker_df = df.xs(ticker, axis=1, level=1).copy()
        
        # 移除可能存在的空行（比如某些资产上市晚，早期日期全为空）
        ticker_df.dropna(how='all', inplace=True)
        
        # 构造保存文件名
        file_name = f"{ticker.lower()}_daily_max.csv"
        ticker_df.to_csv(file_name)
        
        print(f"✅ {ticker} 已保存至: {os.path.abspath(file_name)}")
        print(f"   数据范围: {ticker_df.index[0].date()} 至 {ticker_df.index[-1].date()}")
        print("-" * 30)

if __name__ == "__main__":
    assets = ["SPY", "QQQ"]
    download_and_save_separately(assets)