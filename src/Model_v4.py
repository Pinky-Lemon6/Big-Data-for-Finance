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

torch.set_printoptions(threshold=float('inf')) # torch打印无限制

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

def get_params():
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
    def __init__(self, src_len, tgt_len, d_model, nhead, d_hid, nlayers, batch_size, dropout=0.1):
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
    
    def load_params(self, path):
        params = torch.load(path)
        self.load_state_dict(params)

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

def model_val(model, criterion, val_db, val_loader, mean, var):
    """模型验证"""
    val_loss = 0
    for (X, y) in val_loader:
        X -= mean
        X /= np.sqrt(var)
        y -= mean
        y /= np.sqrt(var)
        src = X.to(device)
        tgt = src[:, 1: ]
        y = y.to(device)
        pred = model(src, tgt)
        loss = criterion(pred, y).item()
        val_loss += loss*X.size(0)
    
    val_loss /= val_db.__len__()
    print('Validation set: Average loss: {:.4f}\n'.format(val_loss))



def model_test(model, criterion, test_db, test_loader, mean, var):
    """TODO:最终模型评估"""
    pass


def main():
    mean, var = get_params() # 获取原数据集均值方差
    train_db = StockDataset(dim_x=9, mode="train")
    test_db = StockDataset(dim_x=9, mode="test")
    train_db, val_db = torch.utils.data.random_split(train_db, [train_db.__len__()-test_db.__len__(), test_db.__len__()]) # 划分训练集、验证集
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=cores)
    val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=False, num_workers=cores)
    test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=False, num_workers=cores)

    my_model = TransformerModel(src_len=9, tgt_len=8, d_model=32, nhead=4, d_hid=128, 
                                nlayers=6, batch_size=batch_size).to(device)
    optimizer = optim.Adam(my_model.parameters(), lr=lr)
    criterion = nn.MSELoss().to(device)

    for epoch in range(epochs):
        # TODO:loss曲线
        my_model.train()
        for batch_idx, (X, y) in enumerate(train_loader):
            # print(batch_idx)
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
                        100. * batch_idx* batch_size / train_db.__len__(), loss.item()))
        
        torch.save(my_model.state_dict(), os.path.join(current_directory, f'../model/transformer_model_epoch{epoch}.pth')) # 保存模型
        # 验证集
        my_model.eval()
        model_val(my_model, criterion, val_db, val_loader, mean, var)

    # 模型评估
    final_epoch = 1
    param_dir = os.path.join(current_directory, f'../model/transformer_model_epoch{final_epoch}.pth')
    my_model.load_params(param_dir)
    my_model.eval(my_model, criterion, test_db, test_loader, mean, var)
    model_test()

    
    

if __name__ == "__main__":
    freeze_support()
    main()