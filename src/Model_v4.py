import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from Dataset_v1 import StockDataset
from torch.utils.data import DataLoader
import os
import numpy as np
from multiprocessing import freeze_support

torch.set_printoptions(threshold=float('inf'))

current_directory = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda:0")
batch_size = 16
epochs = 1000
lr = 0.0001
cores = 7

def get_params():
    """查询数据集的均值方差"""
    if not os.path.exists("./mean_var.txt"):
        import pandas as pd
        data_dir = os.path.join(current_directory, "../Dataset/S&P500/all_stocks_5yr.csv")
        data = pd.read_csv(data_dir)
        close = np.array(data["close"].to_list())
        mean, var = close.mean(), close.var()
        with open("./mean_var.txt", "w") as f:
            f.write(str(mean))
            f.write("\t")
            f.write(str(var))
    with open("./mean_var.txt", "r") as f:
        for line in f:
            mean, var = line.strip().split("\t")
            mean = float(mean)
            var = float(var)
    return mean, var


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, batch_size, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_mat = torch.arange(batch_size).view(-1, 1)
        i_mat = torch.pow(10000, torch.arange(0, d_model, 2).reshape(1, -1)/d_model)
        pe_embedding_table = torch.zeros(batch_size, d_model)
        # 偶数列
        pe_embedding_table[:,  0::2] = torch.sin(pos_mat/i_mat) #broadingcast
        # 奇数列
        pe_embedding_table[:,  1::2] = torch.cos(pos_mat/i_mat)
        self.pe_embedding_table = pe_embedding_table.to(device)

    def forward(self, x):
        x = x + self.pe_embedding_table[:x.size(0), :]
        return self.dropout(x)



class TransformerModel(nn.Module):
    def __init__(self, src_len, tgt_len, d_model, nhead, d_hid, nlayers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding1 = nn.Linear(in_features=src_len, out_features=d_model)
        self.embedding2 = PositionalEncoding(d_model=d_model, batch_size=batch_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        # self.linear = nn.Linear(in_features=d_model, out_features=tgt_len)

        self.embedding3 = nn.Linear(in_features=tgt_len, out_features=d_model)
        self.embedding4 = PositionalEncoding(d_model=d_model, batch_size=batch_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=nlayers)
        self.output_linear = nn.Linear(d_model, 1)  # 将d_model维映射回1维输出
        self.dropout = nn.Dropout(dropout)  # 添加Dropout层以防止过拟合

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding1(src)
        src = self.embedding2(src) # [batch_size, d_model]
        encoder_output = self.transformer_encoder(src, src_key_padding_mask=src_mask)  # 通过Transformer编码器
        # encoder_output = self.linear(encoder_output)

        tgt = self.embedding3(tgt)
        tgt = self.embedding4(tgt)
        output = self.transformer_decoder(tgt, encoder_output, tgt_mask=tgt_mask)  # 通过Transformer解码器
        output = output.transpose(0, 1)
        return output


def main():
    mean, var = get_params() # 获取原数据集均值方差
    train_db = StockDataset(dim_x=9, mode="train")
    test_db = StockDataset(dim_x=9, mode="test")
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=cores)
    test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=False, num_workers=cores)

    my_model = TransformerModel(src_len=9, tgt_len=8, d_model=32, nhead=4, d_hid=128, nlayers=6).to(device)
    optimizer = optim.Adam(my_model.parameters(), lr=lr)
    criterion = nn.MSELoss().to(device)

    for epoch in range(epochs):
        my_model.train()
        for batch_idx, (X, y) in enumerate(train_loader):
            # print(batch_idx)
            if batch_idx == 31:
                pass
            X -= mean
            X /= np.sqrt(var)
            y -= mean
            y /= np.sqrt(var)
            src = X.to(device)
            tgt = src[:, 1: ]
            y = y.to(device)

            pred = my_model(src, tgt)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(X), train_db.__len__(),
                        100. * batch_idx / train_db.__len__(), loss.item()))
        
        my_model.eval()
        test_loss = 0
        for (_, x, y) in test_loader:
            x, y = x.to(device), y.to(device)
            pred = my_model(x)
            loss = criterion(pred, y).item()
            test_loss += loss
        
        test_loss /= test_db.__len__()
        
        print('Validation set: Average loss: {:.4f}\n'.format(test_loss))


if __name__ == "__main__":
    freeze_support()
    main()