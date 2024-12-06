import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
import os
import numpy as np
from multiprocessing import freeze_support
import random
from utils import get_params, model_val
from TransformerModel import TransformerModel, MLP
from sklearn.metrics import mean_absolute_error, r2_score
from DataLoader import StockDataset


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
            # pred = model(src).squeeze(1)
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


def main():
    seed=1234 # 设置随机数种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    current_directory = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 128

    mean, var = get_params(current_directory)
    test_db = StockDataset(dim_x=9, mode="test")
    test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=False, num_workers=7)

    model = TransformerModel(
        src_len=9,
        tgt_len=8,
        d_model=32,
        nhead=4,
        d_hid=128,
        nlayers=3,
        batch_size=batch_size,
        device=device
    ).to(device)
    # model = MLP().to(device)
    # print(model)

    final_epoch = 49
    param_dir = os.path.join(current_directory, f'../../model/transformer_model_epoch{final_epoch}.pth')
    model.load_params(param_dir)
    for name, param in model.named_parameters():
        print(f"Layer: {name}")
        print(f"Size: {param.size()}")
        print(f"Coefficients:\n{param}")
    criterion = nn.MSELoss()
    model_val(model, criterion, test_db, test_loader, mean, var, device)
    mse, mae, rmse, mape = model_eval(model, criterion, test_db, test_loader, mean, var, device)


if __name__ == "__main__":
    freeze_support()
    main()