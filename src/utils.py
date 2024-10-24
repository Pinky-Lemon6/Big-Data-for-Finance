import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from Dataset4web import StockDataset
from torch.utils.data import DataLoader
import os
import numpy as np
from multiprocessing import freeze_support
import random
from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib import pyplot as plt



def get_params(current_directory):
    """查询数据集的均值方差"""
    if not os.path.exists(os.path.join(current_directory, "./mean_var.txt")):
        import pandas as pd
        data_dir = os.path.join(current_directory, "../Dataset/S&P500/all_stocks_5yr.csv")
        data = pd.read_csv(data_dir)
        close = np.array(data["close"].to_list())
        mean, var = close.mean(), close.var()
        with open(os.path.join(current_directory, "./mean_var.txt"), "w") as f:
            f.write(str(mean))
            f.write("\t")
            f.write(str(var))
    with open(os.path.join(current_directory, "./mean_var.txt"), "r") as f:
        for line in f:
            mean, var = line.strip().split("\t")
            mean = float(mean)
            var = float(var)
    return mean, var


def model_val(model, criterion, val_db, val_loader, mean, var, device):
    """模型验证"""
    val_loss = 0
    for (X, y) in val_loader:
        X -= mean
        X /= np.sqrt(var)
        y -= mean
        y /= np.sqrt(var)
        src = X.to(device)
        tgt = src[:, 1: ]
        y = y.unsqueeze(1).to(device)
        pred = model(src, tgt)
        loss = criterion(pred, y).item()
        val_loss += loss*X.size(0)
    
    val_loss /= val_db.__len__()
    print('Validation set: Average loss: {:.4f}\n'.format(val_loss))
    return val_loss



def mean_absolute_percentage_error(y_pred, y_true):
    ret = 0
    for i in range(len(y_true)):
        ret += abs(y_true[i] - y_pred[i]) / (len(y_true) * abs(y_true[i]))
    return ret




def model_eval(model, criterion, test_db, test_loader, mean, var, device):
    test_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            X -= mean
            X /= np.sqrt(var)
            y -= mean
            y /= np.sqrt(var)
            src = X.to(device)
            tgt = src[:, 1: ]
            y = y.to(device)
            pred = model(src, tgt).squeeze(1)
            loss = criterion(pred, y)
            test_loss += loss.item() * X.size(0)
            y_true += y.tolist()
            y_pred += pred.tolist()

            
    mse = test_loss/test_db.__len__()
    mae = mean_absolute_error(y_pred=y_pred, y_true=y_true)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_pred=y_pred, y_true=y_true)
    # r2 = r2_score(y_pred=y_pred, y_true=y_true)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
    # print(f"R2: {r2:.4f}")
    # print(f"Adjusted R2: {adjusted_r2:.4f}")

    return mse, mae, rmse, mape

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    seed=1234 # 设置随机数种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    current_directory = os.path.dirname(os.path.abspath(__file__))

    print("*"*100)
    mean, var = get_params(current_directory=current_directory)
    print("mean: {}, var: {}".format(mean, var))

    print("*"*100)
    y_pred = [1.1, 1.1, 1.1]
    y_true = [1, 1, 1]
    mape = mean_absolute_percentage_error(y_pred=y_pred, y_true=y_true)
    print("mape: {}".format(mape))

    print("*"*100)
    train_losses = [4, 2, 1, 1/2]
    val_losses = [4, 2, 1, 1]
    plot_loss(train_losses=train_losses, val_losses=val_losses)