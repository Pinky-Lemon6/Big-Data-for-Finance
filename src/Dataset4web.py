# 论文中是用前t时刻数据预测后一时刻数据
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os, glob
import pandas as pd

current_directory = os.path.dirname(os.path.abspath(__file__))

class StockDataset(Dataset):
    """输出数据格式[x_{t-s}, x_{t-s+1}, ..., x_{t-1}, x_t], x_{t+1}"""
    def __init__(self, dim_x=9, mode="train"):
        super(StockDataset, self).__init__()
        self.root = os.path.join(current_directory, "../Dataset/S&P500")
        self.dim_x = dim_x
        self.mode = mode

        self.black_lst = ["ADS","AGN","AMZN","AZO","BIIB","BLK","CHTR","CMG","EQIX",
                          "ESS","GOOG","GOOGL","GWW","ISRG","LMT","MTD","ORLY","PCLN",
                          "REGN","SHW","TDG",] # 黑名单，这些股票的均值明显偏离其余
        self.split_date = "2017-01-30" # 所有日期>=split_date的数据均为测试集

        self.file_list_name = os.path.join(self.root, "file_list.csv") # file_list.txt存放每只股票数据的文件名
        self.load_dataset()
        self.read_dataset()

        

    def load_csv(self):
        # 获取所有子数据集文件名
        if not os.path.exists(self.file_list_name):
            file_list = []
            cnt = 0
            for name in os.listdir(os.path.join(self.root, "individual_stocks_5yr/individual_stocks_5yr")):
                if name.endswith(".csv"):
                    if not name.replace("_data.csv", "") in self.black_lst:
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
                    for i in range(0, len(data)-self.dim_x-1, 1):
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

    def read_dataset(self):
        self.data = []
        with open(os.path.join(self.root, self.dataset_name)) as f:
            for line in f:
                self.data.append(line)

                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        line = line.strip("\n")
        x = [float(i) for i in line.split(",")[:-1]]
        y = float(line.split(",")[-1])
        X = torch.tensor(x).to(torch.float32)
        y = torch.tensor(y).to(torch.float32)
        return X, y



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
    