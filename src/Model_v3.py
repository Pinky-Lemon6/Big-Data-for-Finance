import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from Dataset_v2 import StockDataset
import math

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, d_hid, nlayers, dropout, max_len):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.input_linear = nn.Linear(1, d_model)  # 将1维输入映射到d_model维
        
        # Decoder部分
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=nlayers)
        self.output_linear = nn.Linear(d_model, 1)  # 将d_model维映射回1维输出
        self.dropout = nn.Dropout(dropout)  # 添加Dropout层以防止过拟合
        

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 将输入数据从 [batch_size, T, 1] 转换为 [T, batch_size, d_model]
        src = src.transpose(0, 1).squeeze(-1)  # 从 [batch_size, T, 1] 转换为 [T, batch_size]
        src = src.unsqueeze(-1)  # 添加一个维度，使其形状为 [T, batch_size, 1]
        src = self.input_linear(src)  # 从 [T, batch_size, 1] 转换为 [T, batch_size, d_model]
        src = self.pos_encoder(src)  # 添加位置编码
        src = src.transpose(0, 1)  # 转换为 [batch_size, T, d_model]
        
        # 编码器输出
        encoder_output = self.transformer_encoder(src, src_key_padding_mask=src_mask)  # 通过Transformer编码器
        
        # 解码器部分
        tgt = tgt.transpose(0, 1).squeeze(-1)  # 从 [batch_size, N, 1] 转换为 [N, batch_size]
        tgt = tgt.unsqueeze(-1)  # 添加一个维度，使其形状为 [N, batch_size, 1]
        tgt = self.input_linear(tgt)  # 从 [N, batch_size, 1] 转换为 [N, batch_size, d_model]
        tgt = self.pos_encoder(tgt)  # 添加位置编码
        tgt = tgt.transpose(0, 1)  # 转换为 [batch_size, N, d_model]
        
        # 解码器输出
        output = self.transformer_decoder(tgt, encoder_output, tgt_mask=tgt_mask)  # 通过Transformer解码器
        output = output.transpose(0, 1)  # 转换回 [batch_size, N, d_model]
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
d_model = 32  # 嵌入的维度
nhead = 4  # 注意力头的数量
d_hid = 128  # 前馈网络的维度
nlayers = 6  # 编码器层的数量
dropout = 0.1  # Dropout率
max_len = 9  # 最大序列长度

# 创建数据集实例
train_db = StockDataset(mode="train", dim_x=9)
test_db = StockDataset(mode="test", dim_x=9)

# 创建数据加载器
batch_size = 16
train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=False)

# 创建模型实例
model = TransformerModel(d_model, nhead, d_hid, nlayers, dropout, max_len)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# 训练模型
num_epochs = 1000
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for X, y in train_loader:
        # 调整X的形状以匹配模型的输入要求
        X = X.view(-1, 9, 1)  # 重新塑形X以匹配模型的输入要求，形状为[batch_size, seq_len, 1]
        optimizer.zero_grad()
        output = model(X, X, src_mask=None, tgt_mask=None)  # 使用X作为目标输入
        output_last_step = output[-1, :, :]  # 选择输出的最后一个时间步作为预测值
        output_last_step = output_last_step[:,-1]
        loss = criterion(output_last_step, y)  # 确保输出和目标尺寸匹配        
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
    
# 保存模型
torch.save(model.state_dict(), 'transformer_model.pth')