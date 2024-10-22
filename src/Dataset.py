import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

class StockDataset(Dataset):
    """数据集描述：见data_describe.py"""
    def __init__(self, mode="train"):
        super(StockDataset, self).__init__()
        self.data_dir = os.path.join(current_directory, "../Dataset/S&P500/all_stocks_5yr_clean.csv")
        self.len = 619029
        if mode == "train":
            self.len = 488867
            self.pad = 0
        else:
            self.len -= 488867
            self.pad = 488867


    def read_line(self, file_path, line_number):
        with open(file_path, 'r') as file:
            file.seek(0)
            for current_line in range(1, line_number + 1):
                line = file.readline()
                if current_line == line_number:
                    return line
                if not line:
                    return None
                
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        idx += 2 # 首行标题, readfile行号从1开始
        idx += self.pad
        date,open,high,low,close,volume,Name = self.read_line(self.data_dir, idx).strip("\n").split(",")
        open = torch.tensor(float(open)).to(torch.float32)
        high = torch.tensor(float(high)).to(torch.float32)
        low = torch.tensor(float(low)).to(torch.float32)
        close = torch.tensor(float(close)).to(torch.float32)
        volume = torch.tensor(float(volume)).to(torch.float32)
        return date,open,high,low,close,volume,Name
        # return open, high





def main():
    batch_size = 32
    train_db = StockDataset(mode="train")
    test_db = StockDataset(mode="test")
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=False)
    for date,open,high,low,close,volume,Name in train_loader:
        print(date,open,high,low,close,volume,Name)
        break
    for date,open,high,low,close,volume,Name in test_loader:
        print(date,open,high,low,close,volume,Name)
        break

if __name__ == "__main__":
    main()
    