import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 确保导入了模型和数据集
from Dataset4web import StockDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 16


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, batch_size, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_mat = torch.arange(batch_size).view(-1, 1)
        i_mat = torch.pow(10000, torch.arange(0, d_model, 2).reshape(1, -1) / d_model)
        pe_embedding_table = torch.zeros(batch_size, d_model)
        # 偶数列
        pe_embedding_table[:, 0::2] = torch.sin(pos_mat / i_mat)  # broadingcast
        # 奇数列
        pe_embedding_table[:, 1::2] = torch.cos(pos_mat / i_mat)
        self.pe_embedding_table = pe_embedding_table.to(device)

    def forward(self, x):
        x = x + self.pe_embedding_table[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self, src_len, tgt_len, d_model, nhead, d_hid, nlayers, batch_size, dropout=0.1
    ):
        super(TransformerModel, self).__init__()
        self.embedding1 = nn.Linear(in_features=src_len, out_features=d_model)
        self.embedding2 = PositionalEncoding(d_model=d_model, batch_size=batch_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=nlayers
        )
        # self.linear = nn.Linear(in_features=d_model, out_features=tgt_len)

        self.embedding3 = nn.Linear(in_features=tgt_len, out_features=d_model)
        self.embedding4 = PositionalEncoding(d_model=d_model, batch_size=batch_size)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=nlayers
        )
        self.output_linear = nn.Linear(d_model, 1)  # 将d_model维映射回1维输出
        self.dropout = nn.Dropout(dropout)  # 添加Dropout层以防止过拟合

    def load_params(self, path):
        params = torch.load(path)
        self.load_state_dict(params)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding1(src)
        src = self.embedding2(src)  # [batch_size, d_model]
        encoder_output = self.transformer_encoder(
            src, src_key_padding_mask=src_mask
        )  # 通过Transformer编码器
        # encoder_output = self.linear(encoder_output)

        tgt = self.embedding3(tgt)
        tgt = self.embedding4(tgt)
        output = self.transformer_decoder(
            tgt, encoder_output, tgt_mask=tgt_mask
        )  # 通过Transformer解码器
        output = output.transpose(0, 1)
        return output


def get_params():
    # 这个函数需要根据您保存均值和方差的方式进行调整
    current_directory = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_directory, "./mean_var.txt"), "r") as f:
        for line in f:
            mean, var = line.strip().split("\t")
            mean = float(mean)
            var = float(var)
    return mean, var


def model_eval(model, criterion, data_loader, mean, var):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device) - mean
            X = X / np.sqrt(var)
            y = y.to(device) - mean
            src = X[:, :-1]
            tgt = X[:, -1:]
            pred = model(src, tgt)
            loss = criterion(pred, y)
            total_loss += loss.item() * X.shape[0]
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = 9  # 特征数量，根据您的模型进行调整
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    print(f"Average loss: {avg_loss:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"Adjusted R2: {adjusted_r2:.4f}")

    return avg_loss


def main():
    mean, var = get_params()
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
    ).to(device)

    model_path = "../model/transformer_model_epoch1.pth"  # 替换为您的模型路径
    model.load_state_dict(torch.load(model_path, map_location=device))

    criterion = nn.MSELoss()

    model_eval(model, criterion, test_loader, mean, var)


if __name__ == "__main__":
    main()
