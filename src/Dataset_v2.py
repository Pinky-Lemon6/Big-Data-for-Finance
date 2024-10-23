import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os, glob
import pandas as pd
import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))

class StockDataset(Dataset):
    """输出数据格式[x_{t-s}, x_{t-s+1}, ..., x_{t-1}, x_t], x_{t+1}"""
    def __init__(self, dim_x=9, mode="train"):
        super(StockDataset, self).__init__()
        self.root = os.path.join(current_directory, "../Dataset/S&P500")
        self.dim_x = dim_x
        self.mode = mode

        self.split_date = "2017-01-30" # 所有日期>=split_date的数据均为测试集

        self.file_list_name = os.path.join(self.root, "file_list.csv") # file_list.txt存放每只股票数据的文件名
        self.file_list = self.load_csv() # 加载文件列表
        self.load_dataset()

        self.len = -1
        with open(os.path.join(self.root, self.file_list_name), "r") as f:
            for line in f:
                self.len += 1

        # 计算训练集的均值和标准差
        if mode == "train":
            self.mean, self.std = self.calculate_mean_std(self.load_data_for_mean_std())
            # 保存均值和标准差
            np.save(os.path.join(self.root, "mean_std.npy"), (self.mean, self.std))
        else:
            self.mean, self.std = self.load_mean_std()

    def load_csv(self):
        # 获取所有子数据集文件名
        if not os.path.exists(self.file_list_name):
            file_list = []
            cnt = 0
            for name in os.listdir(os.path.join(self.root, "individual_stocks_5yr/individual_stocks_5yr")):
                if name.endswith(".csv"):
                    file_list.append(os.path.normpath(os.path.join(self.root, "individual_stocks_5yr/individual_stocks_5yr", name)))
            print(f"total csv_files: {len(file_list)}")
            with open(self.file_list_name, "w") as f:
                for file_name in file_list:
                    f.write(file_name)
                    f.write("\n")
        file_list = []
        with open(self.file_list_name) as f:
            for line in f:
                file_list.append(line.strip("\n"))
        return file_list


    def standard_date(self, date_str):
        year, month, day = date_str.split("-")
        return "-".join([year, month.zfill(2), day.zfill(2)])
    
    def load_dataset(self):
        if self.mode == "train":
            dataset_name = f"all_stocks_5yr_clean_{self.dim_x}T_train.csv"
            if not os.path.exists(os.path.join(self.root, dataset_name)):
                print(f"{dataset_name} does not exist! creating...")
                self.file_list = self.load_csv()
                for idx, file_name in enumerate(self.file_list):
                    data = pd.read_csv(file_name)
                    data["date"] = data["date"].apply(self.standard_date)
                    data = data[data.date<self.split_date]
                    data = data.sort_values(by = "date", ascending=True).reset_index()
                    for i in range(self.dim_x, len(data)-self.dim_x-1, 1):
                        x = data.loc[i:i+self.dim_x, "close"].to_list() # X=x[:-1], y=x[-1]
                        with open(os.path.join(self.root, dataset_name), "a") as f:
                            f.write(",".join([str(i) for i in x]))
                            f.write("\n")

                    if idx%10 == 0:
                        print(f"{idx}/{len(self.file_list)} files done, {int(100*idx/len(self.file_list))}%")
                        # break
        else:
            dataset_name = f"all_stocks_5yr_clean_{self.dim_x}T_test.csv"
            if not os.path.exists(os.path.join(self.root, dataset_name)):
                print(f"{dataset_name} does not exist! creating...")
                self.file_list = self.load_csv()
                for idx, file_name in enumerate(self.file_list):
                    data = pd.read_csv(file_name)
                    data["date"] = data["date"].apply(self.standard_date)
                    data = data[data.date>=self.split_date]
                    data = data.sort_values(by = "date", ascending=True).reset_index()
                    for i in range(self.dim_x, len(data)-self.dim_x-1, 1):
                        x = data.loc[i:i+self.dim_x, "close"].to_list() # X=x[:-1], y=x[-1]
                        with open(os.path.join(self.root, dataset_name), "a") as f:
                            f.write(",".join([str(i) for i in x]))
                            f.write("\n")

                    if idx%10 == 0:
                        print(f"{idx}/{len(self.file_list)} files done, {int(100*idx/len(self.file_list))}%")
                        # break
        self.dataset_name = dataset_name

    def calculate_mean_std(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return mean, std

    def load_data_for_mean_std(self):
        data_list = []
        for file_name in self.file_list:
            data = pd.read_csv(file_name)
            data["date"] = data["date"].apply(self.standard_date)
            data = data[data.date<self.split_date]
            data = data.sort_values(by = "date", ascending=True).reset_index()
            data_list.append(data["close"].values)
        return np.concatenate(data_list)
    
    def load_mean_std(self):
        mean_std_file = os.path.join(self.root, "mean_std.npy")
        if not os.path.exists(mean_std_file):
            raise FileNotFoundError(f"Mean and standard deviation file not found: {mean_std_file}")
        mean, std = np.load(mean_std_file, allow_pickle=True)
        return mean, std

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        cnt = 0
        with open(os.path.join(self.root, self.dataset_name)) as f:
            for line in f:
                line = line.strip("\n")
                if cnt == idx:
                    x = [float(i) for i in line.split(",")[:-1]]
                    y = float(line.split(",")[-1])
                    # print(f"Original X: {x}")  # 打印原始数据
                    # 归一化处理
                    X = torch.tensor(x, dtype=torch.float32)
                    y = torch.tensor(y, dtype=torch.float32)
                    X = (X - self.mean) / self.std
                    y = (y - self.mean) / self.std
                    # print(f"Normalized X: {X.numpy()}")  # 打印归一化后的数据
                    return X, y
                cnt += 1

def main():
    batch_size=32
    train_db = StockDataset(mode="train", dim_x=9)
    test_db = StockDataset(mode="test", dim_x=9)
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=True)
    for X, y in train_loader:
        print(X.size(), y)
        break
    for X, y in test_loader:
        print(X.size(), y)
        break
    


if __name__ == "__main__":
    main()