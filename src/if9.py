import pandas as pd

# 读取CSV文件
df = pd.read_csv("../Dataset/S&P500/all_stocks_5yr_clean.csv")

# 确保日期列是日期时间格式
df["date"] = pd.to_datetime(df["date"])

# 按天统计每一天的数据条数
daily_counts = df["date"].dt.date.value_counts()

# 将结果导出为CSV文件
daily_counts.to_csv("daily_counts.csv")

# 打印每一天的数据条数
print(daily_counts)
