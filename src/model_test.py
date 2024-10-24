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
from utils import model_eval, get_params
from TransformerModel import TransformerModel


def main():
    seed=1234 # 设置随机数种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    current_directory = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda:0")
    batch_size = 16

    mean, var = get_params(current_directory)
    test_db = StockDataset(dim_x=9, mode="test")
    test_loader = DataLoader(test_db, batch_size=16, shuffle=False, num_workers=7)

    model = TransformerModel(
        src_len=9,
        tgt_len=8,
        d_model=32,
        nhead=4,
        d_hid=128,
        nlayers=6,
        batch_size=batch_size,
        device=device
    ).to(device)

    final_epoch = 1
    param_dir = os.path.join(current_directory, f'../model/transformer_model_epoch{final_epoch}.pth')
    model.load_params(param_dir)
    criterion = nn.MSELoss()
    mse, mae, rmse, mape = model_eval(model, criterion, test_db, test_loader, mean, var, device)


if __name__ == "__main__":
    freeze_support()
    main()