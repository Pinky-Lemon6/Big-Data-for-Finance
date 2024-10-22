import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset1 import StockDataset
import math

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_encoder_layers, forward_expansion, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        # 注意：不再在这里创建位置编码，我们将在forward方法中动态创建
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=forward_expansion * embed_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.Linear(embed_dim, 1)

    def _create_positional_encoding(self, embed_dim, sequence_length):
        pe = torch.zeros(sequence_length, embed_dim)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    # def forward(self, src):
    #     src = self.embedding(src)  # src shape: [batch_size, sequence_length, embed_dim]
    #     sequence_length = src.size(1)
    #     src = src * math.sqrt(self.embedding.weight.shape[1])
    #     # 动态创建位置编码，确保其长度与输入序列相匹配
    #     positional_encoding = self._create_positional_encoding(src.size(2), sequence_length)
    #     src = src + positional_encoding[:sequence_length, :].unsqueeze(0)
    #     output = self.transformer_encoder(src)
    #     output = self.decoder(output)
    #     return output[:, -1, :]  # 只返回最后一个时间步的输出
    
    def forward(self, src):
        batch_size, sequence_length, input_dim = src.size()
        src = self.embedding(src)  # src shape: [batch_size, sequence_length, embed_dim]
        src = src * math.sqrt(self.embedding.weight.shape[1])
        positional_encoding = self._create_positional_encoding(src.size(2), sequence_length)
        src = src + positional_encoding[:sequence_length, :].unsqueeze(0)  # Add positional encoding
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output[:, -1, :]  # 只返回最后一个时间步的输出


# 超参数设置
input_dim = 5  # 特征数量：开盘价、最高价、最低价、收盘价、成交量
embed_dim = 32
num_heads = 4
num_encoder_layers = 3
forward_expansion = 2
dropout = 0.1
learning_rate = 0.001

# 创建模型实例
model = TransformerModel(input_dim, embed_dim, num_heads, num_encoder_layers, forward_expansion, dropout)

# 数据加载
batch_size = 32
sequence_length = 10
train_db = StockDataset(mode="train")
test_db = StockDataset(mode="test")
train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True,drop_last=True)
test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=False,drop_last=True)

# 训练模型
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10
# for epoch in range(num_epochs):
#     for i, (date, open, high, low, close, volume, Name) in enumerate(train_loader):
#         optimizer.zero_grad()
        
#         # 将特征合并为一个张量，并增加一个维度以匹配模型的输入
#         features = torch.stack((open, high, low, close, volume), dim=1)  # 使用stack而不是cat
        
#         # 目标是下一个交易日的收盘价，这里我们使用shift来创建标签
#         labels = close.view(-1, 1)
        
#         outputs = model(features)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         if i % 100 == 0:
#             print(f'Epoch {epoch+1}, Step {i+1}, Loss: {loss.item()}')

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        # features = torch.stack((batch["open"], batch["high"], batch["low"], batch["close"], batch["volume"]), dim=1)
        features = torch.randn(batch_size, sequence_length, input_dim)
        labels = batch["close"].view(-1, 1)
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 保存模型
torch.save(model.state_dict(), 'transformer_model_sp500.pth')

# # 测试模型
# model.eval()
# with torch.no_grad():
#     total_loss = 0
#     for date, open, high, low, close, volume, Name in test_loader:
#         features = torch.stack((open, high, low, close, volume), dim=1)
#         labels = close.view(-1, 1)
#         outputs = model(features)
#         loss = criterion(outputs, labels)
#         total_loss += loss.item()
#     print(f'Test Loss: {total_loss / len(test_loader)}')