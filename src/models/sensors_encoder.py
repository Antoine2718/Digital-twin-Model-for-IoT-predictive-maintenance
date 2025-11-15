import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.out_dim = hidden_dim

    def forward(self, x):
        out, _ = self.lstm(x)
        return out  # [B, T, H]

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_dim = d_model

    def forward(self, x):
        x = self.proj(x)
        mask = None
        out = self.encoder(x, mask)  # [B, T, d_model]
        return out
