import pandas as pd
import os

# 1. 定义文件路径
file_path = "QuantData/Stock/massive/spot/QQQ_1m.csv"

# 确保文件存在
if os.path.exists(file_path):
    print(f"正在读取文件: {file_path}")
    df = pd.read_csv(file_path)
    
    # 2. 核心逻辑 1：直接通过纳秒整除 1,000,000 得到毫秒时间戳
    df['open_time_ms_utc'] = df['window_start'] // 1000000
    
    # 3. 核心逻辑 2：将 window_start（纳秒）转换为指定的日期时间格式字符串 'YYYY-MM-DD HH:MM:SS'
    # unit='ns' 表示输入是纳秒，utc=True 确保按 UTC 时区解析，strftime 负责去掉末尾的时区后缀（如 +00:00）
    df['open_time_date_utc'] = pd.to_datetime(df['window_start'], unit='ns', utc=True).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 4. 保存回原 CSV 文件
    df.to_csv(file_path, index=False)
    print(f"✅ 处理完成！已成功添加 'open_time_ms_utc' 和 'open_time_date_utc' 列并覆盖原文件。")
    
    # 打印前几行预览
    print("\n数据预览:")
    preview_cols = ['ticker', 'window_start', 'datetime_utc', 'open_time_ms_utc', 'open_time_date_utc']
    print(df[preview_cols].head())

else:
    print(f"❌ 错误：未找到文件，请检查路径是否正确: {file_path}")