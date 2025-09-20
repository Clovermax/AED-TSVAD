import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout=0.0, max_len=4000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(0, num_hiddens, 2, dtype=torch.float32)
        div_term = torch.exp(i * (-np.log(10000.0) / num_hiddens))

        pe = torch.zeros(max_len, num_hiddens)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, num_hiddens)

        self.register_buffer('P', pe)

    def forward(self, X):
        X = X + self.P[:, :X.size(1)]
        return self.dropout(X)

    def forward_pe(self, X):
        """Returns only the positional encoding"""
        return self.P[:, :X.size(1)]
