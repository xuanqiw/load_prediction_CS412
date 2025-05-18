import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :x.size(2)]  # 修改这里以支持正确的维度

class TransformerModel(nn.Module):
    def __init__(self, input_dim=6, d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.2):
        super(TransformerModel, self).__init__()

        # 特征投影层 - 确保输出维度与d_model匹配
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),  # 直接投影到d_model维度
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.pos_encoder = PositionalEncoding(d_model)
        self.norm_first = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model//2),  # 从d_model降维
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, src):
        src = self.input_projection(src)  # [batch_size, seq_len, d_model]
        src = self.pos_encoder(src)
        src = self.norm_first(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output).squeeze(-1)
        return output