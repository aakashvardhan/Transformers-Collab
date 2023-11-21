# We will implement the multi head attention module here in a modular way

# Importing the libraries
import torch
import torch.nn as nn
import math
# from lightning import LightningModule
import numpy as np
import torch.nn.functional as F

# Creating the class for the multi head attention module that we can use for GPT, BERT and ViT. 


# ==================== GPT Multi Head Attention Block ==================== #
class GPTAttentionBlock(nn.Module):
    """
    One head of the self-attention layer
    """
    
    def __init__(self, head_size, num_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(num_embed, head_size, bias=False)
        self.query = nn.Linear(num_embed, head_size, bias=False)
        self.value = nn.Linear(num_embed, head_size, bias=False)
        # tril is a lower triangular matrix. it is not a parameter, so it is not trainable
        # it is assigned to the model using register_buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # B = batch size, T = block size, C = num_embed
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention score
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**(-0.5)
        # tril is a lower triangular matrix, used to mask the future positions
        # this is to prevent the model from cheating by looking at the future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # weighted aggregation of values
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class GPT_MultiHeadAttention(nn.Module):
    """
    Multiple heads of the self-attention layer in parallel
    """
    def __init__(self, num_heads, head_size, num_embed, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            GPTAttentionBlock(head_size=head_size, 
                          num_embed=num_embed, 
                          block_size=block_size, 
                          dropout=dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(num_embed, num_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # output of the self-attention
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # apply the linear projection layer
        out = self.dropout(self.proj(out))
        return out

# ==================== BERT Multi Head Attention Block ==================== #
class BERT_MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, out_dim, dropout=0.1):
        super().__init__()

        # self.q_linear = nn.Linear(out_dim, out_dim)
        # self.k_linear = nn.Linear(out_dim, out_dim)
        # self.v_linear = nn.Linear(out_dim, out_dim)

        self.linear = nn.Linear(out_dim, out_dim*3)

        self.n_heads = n_heads
        self.out_dim = out_dim
        self.out_dim_per_head = out_dim // n_heads
        self.out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, t):
        return t.reshape(t.shape[0],
                         -1,
                         self.n_heads,
                         self.out_dim_per_head)

    @staticmethod
    def attention(q, k, v, mask=None, dropout=None):
        scores = q.matmul(k.transpose(-2, -1))
        scores /= math.sqrt(q.shape[-1]) #expensive in terms of time complexity

        #mask
        scores = scores if mask is None else scores.masked_fill(mask == 0, -1e3)

        scores = F.softmax(scores, dim=-1)
        scores = dropout(scores) if dropout is not None else scores
        output = scores.matmul(v)
        return output, scores

    def forward(self, x, y=None, mask=None):
        # in decoder, y comes from encoder. In encoder, y=x
        y = x if y is None else y
        # out_dim = embed size L
        qkv = self.linear(x) # (batch_size, seq_len, out_dim*3)
        q = qkv[:, :, :self.out_dim] # (batch_size, seq_len, out_dim)
        k = qkv[:, :, self.out_dim:2*self.out_dim] # (batch_size, seq_len, out_dim)
        v = qkv[:, :, 2*self.out_dim:] # (batch_size, seq_len, out_dim)

        #break into n_heads
        q, k, v = [self.split_heads(t) for t in [q, k, v]] # (batch_size, seq_len, n_heads, out_dim_per_head)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]] # (batch_size, n_heads, seq_len, out_dim_per_head)

        # n_heads => attention => merge the heads => mix information
        scores, attn = self.attention(q, k, v, mask, self.dropout) # (batch_size, n_heads, seq_len, out_dim_per_head)
        scores = scores.transpose(1, 2).contiguous().view(scores.shape[0], -1, self.out_dim) # (batch_size, seq_len, out_dim)
        out = self.out(scores) # (batch_size, seq_len, out_dim)
        return out

# ==================== ViT Multi Head Self-Attention Block ==================== #

# 1. Create a class that inherits from nn.Module
class VIT_MultiheadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short).
    """
    # 2. Initialize the class with hyperparameters from Table 1
    def __init__(self,
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0): # doesn't look like the paper uses any dropout in MSABlocks
        super().__init__()

        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. Create the Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True) # does our batch dimension come first?

    # 5. Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, # query embeddings
                                             key=x, # key embeddings
                                             value=x, # value embeddings
                                             need_weights=False) # do we need the weights or just the layer outputs?
        return attn_output

# ==================== Modular Attention Parent Class ==================== # 


class Attention(nn.Module):
    def __init__(self, mode, **kwargs):
        super().__init__()
        self.mode = mode
        # ignore capitalization
        self.mode = self.mode.lower()
        
        if self.mode not in ['bert', 'gpt', 'vit']:
            raise ValueError("Mode must be 'GPT', 'BERT' or 'ViT'")

        
        if self.mode == 'gpt':
            # GPT-specific parameters
            num_heads = kwargs.get('num_heads')
            head_size = kwargs.get('head_size')
            num_embed = kwargs.get('num_embed')
            block_size = kwargs.get('block_size')
            dropout = kwargs.get('dropout', 0.1)
            
            self.attention = GPT_MultiHeadAttention(num_heads, head_size, num_embed, block_size, dropout)
            
        elif self.mode == 'bert':
            # BERT-specific parameters
            n_heads = kwargs.get('n_heads')
            out_dim = kwargs.get('out_dim')
            dropout = kwargs.get('dropout', 0.1)
            
            self.attention = BERT_MultiHeadAttention(n_heads, out_dim, dropout)
            
        elif self.mode == 'vit':
            # ViT-specific parameters
            embedding_dim = kwargs.get('embedding_dim',768)
            num_heads = kwargs.get('num_heads',12)
            attn_dropout = kwargs.get('attn_dropout', 0)
            
            self.attention = VIT_MultiheadSelfAttentionBlock(embedding_dim, num_heads, attn_dropout)
            
    def forward(self, x, y=None, mask=None):
        if self.mode == 'gpt':
            return self.attention(x)
        elif self.mode == 'bert':
            return self.attention(x, y, mask)
        elif self.mode == 'vit':
            return self.attention(x)