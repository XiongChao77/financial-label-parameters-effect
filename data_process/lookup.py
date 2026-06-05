import pandas as pd
from pathlib import Path
import time

target = 'QQQ'

data_dir = Path("financial-label-parameters-effect/data_process/massive/day")
output_path = Path(f"{target.lower()}_1d.csv")
files = sorted(data_dir.glob("*.csv.gz"))
target_data_list = []

print(time.time())
for file in files:
    print(f"Processing: {file.name}")
    
    df = pd.read_csv(file)
    # all_stock = sorted(df["ticker"].unique())
    target_candidates = ["QQQ", "QQQQ"]
    target_data = df[df["ticker"].isin(target_candidates)].copy()
    if not target_data.empty:
        target_data_list.append(target_data)

print(time.time())
target_data_all = pd.concat(target_data_list, ignore_index=True)
target_data_all = target_data_all.sort_values("window_start").reset_index(drop=True)

# 转换为日期时间格式（必须在检查缺失前转换，因为检查需要用到 datetime 对象）
target_data_all["datetime_utc"] = pd.to_datetime(target_data_all["window_start"], unit="ns", utc=True)

# ==================== 新增的检查函数 ====================
def check_missing_periods(df, datetime_col="datetime_utc"):
    """
    检查 DataFrame 中的日期序列是否缺失了某些年份或月份
    """
    if df.empty:
        print("警告: 数据集为空，无法检查缺失。")
        return

    # 1. 获取数据的实际最小和最大时间
    min_date = df[datetime_col].min()
    max_date = df[datetime_col].max()
    
    print("\n" + "="*40)
    print(f"数据时间范围: {min_date.strftime('%Y-%m')} 至 {max_date.strftime('%Y-%m')}")
    
    # 2. 生成理论上完整的“年-月”时间序列 (以月为单位)
    expected_months = pd.date_range(start=min_date, end=max_date, freq='MS').to_period('M')
    
    # 3. 提取数据中实际存在的“年-月”
    actual_months = pd.PeriodIndex(df[datetime_col].dt.to_period('M').unique())
    
    # 4. 找出缺失的月份
    missing_months = expected_months.difference(actual_months)
    
    # 5. 打印检查结果
    if len(missing_months) == 0:
        print("✅ 检查通过：在数据范围内，没有缺失任何月份或年份！")
    else:
        print(f"❌ 警告：发现缺失！在当前时间范围内，共缺失了 {len(missing_months)} 个月份：")
        
        # 按年份分组归纳，方便查看是不是整年缺失
        missing_dict = {}
        for m in missing_months:
            missing_dict.setdefault(m.year, []).append(m.month)
            
        for year, months in sorted(missing_dict.items()):
            if len(months) == 12:
                print(f"  - 缺失整年: {year} 年")
            else:
                months_str = ", ".join(f"{m:02d}月" for m in sorted(months))
                print(f"  - {year} 年缺失月份: {months_str}")
    print("="*40 + "\n")

# 执行检查
check_missing_periods(target_data_all)
# ========================================================

target_data_all.to_csv(output_path, index=False)

print("save to:", output_path)
print("data length:", len(target_data_all))
print(target_data_all.head())
print(target_data_all.tail())
print(time.time())