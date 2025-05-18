import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim=6, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 层：输入形状 (batch, seq_len, input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        # 全连接解码器，将 LSTM 输出映射到预测值（每个时刻输出一个标量）
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, src):
        # src: (batch, seq_len, input_dim)
        batch_size = src.size(0)
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=src.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=src.device)

        # LSTM 得到输出：形状 (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(src, (h0, c0))
        # 对每个时刻的输出使用解码器：结果形状 (batch, seq_len, 1)
        out = self.decoder(lstm_out)
        # squeeze 最后一维，输出 (batch, seq_len)
        return out.squeeze(-1)


