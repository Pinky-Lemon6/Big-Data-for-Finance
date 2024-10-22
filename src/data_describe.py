# total days: 1259
# total_rows of head(1000): 488877
# rate: 0.7897341044197467

import os
current_directory = os.path.dirname(os.path.abspath(__file__))


import pandas as pd
pd.set_option('display.max_rows', None)


data_dir = os.path.join(current_directory, "../Dataset/S&P500/all_stocks_5yr.csv")
data_out_dir = os.path.join(current_directory, "../Dataset/S&P500/all_stocks_5yr_clean.csv")
data = pd.read_csv(data_dir).dropna()
# print(data.head())
data["year"] = data["date"].apply(lambda x: int(x.split("-")[0]))
data["month"] = data["date"].apply(lambda x: int(x.split("-")[1]))
data["day"] = data["date"].apply(lambda x: int(x.split("-")[2]))
data_out = data.sort_values(by = ["year", "month", "day"], ascending=True).reset_index()
data_out = data_out[["date", "open", "high", "low", "close", "volume", "Name"]]
if not os.path.isdir(data_out_dir):
    data_out.to_csv(data_out_dir, index=False)
x = data.groupby(["year", "month", "day", "date"]).size().reset_index()
x.columns = ["year", "month", "day", "date", "cnt"]
x = x.sort_values(by = ["year", "month", "day"], ascending=True)
# print(x.head())
if not os.path.isdir("data_describe.txt"):
    with open("data_describe.txt", "w") as f:
        for i in range(x.shape[0]):
            f.write(x.loc[i, "date"])
            f.write("\t")
            f.write(str(x.loc[i, "cnt"]))
            f.write("\n")

head1000 = x["cnt"].head(1000).sum()
days1000 = x.loc[1000, ["year", "month", "day"]]
total = x["cnt"].sum()
print(f"total days: {x.shape[0]}, total_columns: {total}")
print(f"1000 days is: {days1000}")
print(f"total_rows of head(1000): {head1000}")
print(f"rate: {head1000/total}")