# Importing Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import Attention 
from models.feed_forward import FeedForward as FF

# ==================== BERT Encoder Block ==================== #
class EncoderLayer(nn.Module):
    def __init__(self, n_heads, inner_transformer_size, inner_ff_size, dropout=0.1):
        super().__init__()
        mode = 'bert'
        self.mha = Attention(mode='BERT', n_heads=n_heads, out_dim=inner_transformer_size, dropout=dropout)
        self.ff = FF(mode=mode,inp_dim=inner_transformer_size, inner_dim=inner_ff_size, dropout=dropout)
        self.norm1 = nn.LayerNorm(inner_transformer_size)
        self.norm2 = nn.LayerNorm(inner_transformer_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.mha(x2, mask=mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x