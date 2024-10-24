import torch
from torch import nn
import numpy as np
import random

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
        output = self.output_linear(output)
        # print(output.size())
        # output = output.transpose(0, 1)
        return output
    
    
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
        x = x + self.pe_embedding_table[:x.size(0), :].detach()
        return self.dropout(x)
    
    
    
       
if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # 模型参数
    src_len = 10
    tgt_len = 9
    d_model = 64
    nhead = 4
    d_hid = 128
    nlayers = 2
    batch_size = 32

    # 创建模型实例
    model = TransformerModel(src_len, tgt_len, d_model, nhead, d_hid, nlayers, batch_size)

    # 创建随机输入数据
    src = torch.rand(batch_size, src_len)
    tgt = torch.rand(batch_size, tgt_len)

    # 前向传播
    output = model(src, tgt)

    # 打印输出形状
    print("Output shape:", output.shape)