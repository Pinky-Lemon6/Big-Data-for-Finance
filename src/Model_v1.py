import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from Dataset_v1 import StockDataset

import math

# Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.encoder1 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, d_hid), nlayers)
        self.encoder2 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, d_hid), nlayers)
        self.encoder3 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, d_hid), nlayers)
        self.encoder4 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, d_hid), nlayers)
        self.encoder5 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, d_hid), nlayers)
        self.encoder6 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, d_hid), nlayers)
        self.decoder1 = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, nhead, d_hid), nlayers)
        self.decoder2 = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, nhead, d_hid), nlayers)
        self.decoder3 = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, nhead, d_hid), nlayers)
        self.decoder4 = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, nhead, d_hid), nlayers)
        self.decoder5 = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, nhead, d_hid), nlayers)
        self.decoder6 = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, nhead, d_hid), nlayers)
        
        

        

    def forward(self, src, src_mask):
         src = self.pos_encoder(src)
         src2 = self.encoder(src)
         src2 = self.encoder_norm(src2)
         output = self.output_layer(src2)
         return output


# 创建数据集实例
train_db = StockDataset(mode="train", dim_x=9)
test_db = StockDataset(mode="test", dim_x=9)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=False)

# 实例化模型
model = TransformerModel(
    ntoken=1,  # 输出维度为1，因为我们预测的是下一个时间点的收盘价
    d_model=1,  # 嵌入的维度
    nhead=8,  # 注意力头的数量
    d_hid=2048,  # 前馈网络的维度
    nlayers=6,  # 编码器和解码器层的数量
    dropout=0.1
)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    for X, y in train_loader:
        # 调整X的形状以匹配模型的输入要求
        X = X.unsqueeze(1)  # 添加序列长度的维度
        y = y.unsqueeze(1)  # 添加序列长度的维度
        optimizer.zero_grad()
        output = model(X, src_mask=None)  # 根据您的模型调整 src_mask
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')

# # 评估模型
# model.eval()
# total_loss = 0
# with torch.no_grad():
#     for X, y in test_loader:
#         # 调整X的形状以匹配模型的输入要求
#         X = X.unsqueeze(1)  # 添加序列长度的维度
#         y = y.unsqueeze(1)  # 添加序列长度的维度
#         output = model(X, src_mask=None)
#         loss = criterion(output, y)
#         total_loss += loss.item()
#     print(f'Average Test Loss: {total_loss / len(test_loader)}')

# 保存模型
torch.save(model.state_dict(), 'transformer_model.pth')