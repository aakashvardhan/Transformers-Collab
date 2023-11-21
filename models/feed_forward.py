# Importing the libraries
import torch
import torch.nn as nn
import math
# from lightning import LightningModule
import numpy as np
import torch.nn.functional as F

# ==================== BERT Feed Forward Block ==================== #
class BERT_FeedForwardBlock(nn.Module):
    def __init__(self, inp_dim, inner_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(inp_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, inp_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # inp => inner => relu => dropout => inner => out
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
# ==================== GPT Feed Forward Block ==================== #
class GPT_FeedForward(nn.Module):
    """
    a simple feed-forward layer followed by relu activation
    """
    
    def __init__(self, num_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            # in the original paper
            # authors are using the size of the ffwd layer 2048
            # and the output of the transformer layer is 512
            # so we apply the same factor of 4 to the hidden layer size
            nn.Linear(num_embed, num_embed * 4),
            nn.ReLU(),
            # apply linear projection to restore the original size
            nn.Linear(num_embed * 4, num_embed),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)
    
# ==================== Feed Forward Parent Class ==================== #

class FeedForward(nn.Module):
    def __init__(self, mode, **kwargs):
        super().__init__()
        self.mode = mode
        # ignore capitalization
        self.mode = self.mode.lower()
        
        if self.mode not in ['bert', 'gpt']:
            raise ValueError("Mode must be 'GPT', 'BERT' or 'ViT'")
        
        # Provide default values for each parameter
        default_dropout = 0.1  # Set a sensible default value for dropout
        
        if self.mode == 'bert':
            inp_dim = kwargs.get('inp_dim')
            inner_dim = kwargs.get('inner_dim')
            dropout = kwargs.get('dropout', default_dropout)
            self.ff = BERT_FeedForwardBlock(inp_dim, inner_dim, dropout)
        elif self.mode == 'gpt':
            num_embed = kwargs.get('num_embed')
            dropout = kwargs.get('dropout', default_dropout)
            self.ff = GPT_FeedForward(num_embed, dropout)
            
    def forward(self, x):
        return self.ff(x)
        