import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import Attention
from models.feed_forward import FeedForward as FF



class TransformerBlock(nn.Module):
    """
    This class will group the multi-head attention and the feed-forward layer
    so that we can copy in transformer blocks
    """
    
    def __init__(self, num_heads, block_size, num_embed, dropout):
        super().__init__()
        head_size = num_embed // num_heads
        self.sa = Attention(
            mode='GPT',
            num_heads=num_heads,
            head_size=head_size,
            num_embed=num_embed,
            block_size=block_size,
            dropout=dropout,
        )
        self.ffwd = FF(num_embed=num_embed, dropout=dropout)
        # layer normalization
        self.ln1 = nn.LayerNorm(num_embed)
        self.ln2 = nn.LayerNorm(num_embed)
        
    def forward(self, x):
        # 'x +' is a residual connection
        # it helps with optimization
        # also we apply layer norm before self-attention
        # and after the feed-forward layer (reshuffle from the original paper)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x