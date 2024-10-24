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
from utils import get_params, model_val, plot_loss
from TransformerModel import TransformerModel
from StockDataset import StockDataset


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
    epochs = 2
    lr = 0.0001
    cores = 7

    mean, var = get_params(current_directory) # 获取原数据集均值方差
    train_db = StockDataset(dim_x=9, mode="train")
    test_db = StockDataset(dim_x=9, mode="test")
    train_db, val_db = torch.utils.data.random_split(train_db, [train_db.__len__()-test_db.__len__(), test_db.__len__()]) # 划分训练集、验证集
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=cores)
    val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=False, num_workers=cores)

    my_model = TransformerModel(src_len=9, tgt_len=8, d_model=32, nhead=4, d_hid=128, 
                                nlayers=6, batch_size=batch_size, device=device).to(device)
    optimizer = optim.SGD(my_model.parameters(), lr=lr)
    criterion = nn.MSELoss().to(device)

    train_losses = []  # 用于存储每个epoch的训练损失
    val_losses = []  # 用于存储每个epoch的验证损失

    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_valid_loss = 0

        my_model.train()        
        for batch_idx, (X, y) in enumerate(train_loader):
            # print(batch_idx)
            X -= mean
            X /= np.sqrt(var)
            y -= mean
            y /= np.sqrt(var)
            src = X.to(device)
            tgt = src[:, 1: ]
            y = y.unsqueeze(1).to(device)

            pred = my_model(src, tgt)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * X.size(0)

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(X), train_db.__len__(),
                        100. * batch_idx* batch_size / train_db.__len__(), loss.item()))
        
        epoch_train_loss /= train_db.__len__()
        train_losses.append(epoch_train_loss)
        torch.save(my_model.state_dict(), os.path.join(current_directory, f'../model/transformer_model_epoch{epoch}.pth')) # 保存模型
        # 验证集
        my_model.eval()
        epoch_valid_loss = model_val(my_model, criterion, val_db, val_loader, mean, var, device)
        val_losses.append(epoch_valid_loss)
    plot_loss(train_losses=train_losses, val_losses=val_losses)


if __name__ == "__main__":
    freeze_support()
    main()