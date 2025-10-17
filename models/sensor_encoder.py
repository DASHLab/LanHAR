import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  

    def forward(self, x):  
        x = x + self.pe[:x.size(1)].unsqueeze(0).to(x.device)  
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    def forward(self, src, src_key_padding_mask=None):  
        x = self.input_linear(src)              # [B, T, D]
        x = self.positional_encoding(x)         # [B, T, D]
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, T, D]

        return x

class TimeSeriesClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, num_classes):
        super().__init__()
        self.backbone = TimeSeriesTransformer(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src, src_key_padding_mask=None):
        h = self.backbone(src, src_key_padding_mask) 
        h = h.mean(dim=1)        
        return self.classifier(h) 



