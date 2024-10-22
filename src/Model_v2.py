import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from Dataset_v1 import StockDataset
import math

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, d_hid, nlayers, dropout=0.5, max_len=5000):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.input_linear = nn.Linear(1, d_model)  # 将1维输入映射到d_model维

    def forward(self, src, src_mask=None):
        # 将输入数据从 [batch_size, seq_len, 1] 转换为 [seq_len, batch_size, d_model]
        src = src.transpose(0, 1).squeeze(-1)  # 从 [batch_size, seq_len, 1] 转换为 [seq_len, batch_size]
        src = src.unsqueeze(-1)  # 添加一个维度，使其形状为 [seq_len, batch_size, 1]
        src = self.input_linear(src)  # 从 [seq_len, batch_size, 1] 转换为 [seq_len, batch_size, d_model]
        src = self.pos_encoder(src)  # 添加位置编码
        src = src.transpose(0, 1)  # 转换为 [seq_len, batch_size, d_model]
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)  # 通过Transformer编码器
        output = output.transpose(0, 1)  # 转换回 [batch_size, seq_len, d_model]
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # 修改为在第1维度上添加一个维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :, :]  # 确保位置编码的长度与输入序列的长度匹配
        return self.dropout(x)


# 参数设置
d_model = 1  # 嵌入的维度
nhead = 1  # 注意力头的数量
d_hid = 2048  # 前馈网络的维度
nlayers = 6  # 编码器层的数量
dropout = 0.1  # Dropout率
max_len = 9  # 最大序列长度

# 创建数据集实例
train_db = StockDataset(mode="train", dim_x=9)
test_db = StockDataset(mode="test", dim_x=9)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=False)

# 创建模型实例
model = TransformerModel(d_model, nhead, d_hid, nlayers, dropout, max_len)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # 训练模型
# num_epochs = 10
# model.train()
# for epoch in range(num_epochs):
#     for X, y in train_loader:
#         # 调整X的形状以匹配模型的输入要求
#         X = X.unsqueeze(1)  # 添加序列长度的维度
#         y = y.unsqueeze(1)  # 添加序列长度的维度
#         optimizer.zero_grad()
#         output = model(X, src_mask=None)  # 根据您的模型调整 src_mask
#         loss = criterion(output, y)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch}, Loss: {loss.item()}')

# # 假设输入数据是[16, 9, 1]的张量
# input_data = torch.randn(32, 9, 1)  # 随机生成模拟数据

# # 前向传播
# output = model(input_data)
# print(output.shape)  # 应该输出 torch.Size([16, 9, 1])

# 训练模型
num_epochs = 1000
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for X, y in train_loader:
        # 调整X的形状以匹配模型的输入要求
        X = X.squeeze(1)  # 确保X是3D张量
        # y = y.squeeze(1)  # 确保y是3D张量
        optimizer.zero_grad()
        output = model(X, src_mask=None)  # 根据您的模型调整 src_mask
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')