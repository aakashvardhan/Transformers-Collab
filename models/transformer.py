import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bert_pos_embedding import PositionalEncoding as pe
import models.bert_encoder as bert_encoder
from models.gpt_transformer_block import TransformerBlock as tb

# ==================== BERT Transformer Block ==================== #

class BERT(nn.Module):
    def __init__(self, n_code, n_heads, embed_size, inner_ff_size, n_embeddings, seq_len, dropout=0.1):
        super().__init__()

        #model input
        self.embeddings = nn.Embedding(n_embeddings, embed_size)
        self.pe = pe(embed_size, seq_len)

        #backbone
        encoders = []
        for i in range(n_code):
            encoders += [bert_encoder.EncoderLayer(n_heads, embed_size, inner_ff_size, dropout)]
        self.encoders = nn.ModuleList(encoders)

        #language model
        self.norm = nn.LayerNorm(embed_size)
        self.linear = nn.Linear(embed_size, n_embeddings, bias=False)

    def forward(self, x):
        x = self.embeddings(x)
        x = x + self.pe(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = self.norm(x)
        x = self.linear(x)
        return x
    
# ==================== GPT ==================== #

class GPT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # a simple lookup table that stores embeddings of a fixed dictionary and size
        # each token directly reads off the logits for the next token from a lookup table
        # see more details here: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.vocab_size = kwargs.get('vocab_size', 100)
        self.num_embed = kwargs.get('num_embed', 32)
        self.block_size = kwargs.get('block_size', 8)
        self.num_heads = kwargs.get('num_heads', 4)
        self.num_layers = kwargs.get('num_layers', 4)
        self.dropout = kwargs.get('dropout', 0.2)
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        # each token reads the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.num_embed)
        # each position from 0 to block_size - 1 has a corresponding position embedding
        self.position_embedding_table = nn.Embedding(self.block_size, self.num_embed)
        # transformer blocks
        self.blocks = nn.Sequential(*[
            tb(
                num_heads=self.num_heads,
                block_size=self.block_size,
                num_embed=self.num_embed,
                dropout=self.dropout,
            )
            for _ in range(self.num_layers)
        ])
        # we add the layer norm before the linear layer
        self.ln_f = nn.LayerNorm(self.num_embed)
        self.lm_head = nn.Linear(self.num_embed, self.vocab_size)
        
    def forward(self, idx, targets=None):
        # idx: (B, T)
        # targets: (B, T)
        B, T = idx.shape
        # the token emb is (B, T, C) where C is the num_embed
        token_emb = self.token_embedding_table(idx)
        # the position emb is (T, C)
        posit_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        # we add the two embeddings together
        
        x = token_emb + posit_emb
        # apply one head of the self-attention layer
        x = self.blocks(x)
        # (B, T, vocab_size)
        logits = self.lm_head(x)
        # compute the loss
        if targets != None:
            # cross_entropy accepts inputs in a (batch size, num_classes) shape
            # so we reshape the logits
            # (batch size*time, dim_vocab), time = block_size
            B, T, C = logits.shape
            logits = torch.reshape(logits, (B*T, C))
            targets = torch.reshape(targets, (B*T,))
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens:int, block_size:int):
        # idx: (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the context to last block_size tokens
            idx_crop = idx[:, -block_size:]
            # get the predictions for the next token
            logits, loss = self.forward(idx_crop)
            # focus on the last time step
            logits = logits[:, -1, :] # (B, C)
            # get the probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution 
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=-1) # (B, T+1)
        return idx