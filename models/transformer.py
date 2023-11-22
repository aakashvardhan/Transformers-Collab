import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bert_pos_embedding import PositionalEncoding as pe
import models.bert_encoder as bert_encoder
from models.gpt_transformer_block import TransformerBlock as tb
from models.vit_patch_embedding import PatchEmbedding
from models.vit_transformer_block import TransformerEncoderBlock

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
    
# ==================== ViT ==================== #

        
# 1. Create a ViT class that inherits from nn.Module
class ViT(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 img_size:int=224, # Training resolution from Table 3 in ViT paper
                 in_channels:int=3, # Number of channels in input image
                 patch_size:int=16, # Patch size
                 num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers
                 embedding_dropout:float=0.1, # Dropout for patch and position embeddings
                 num_classes:int=1000): # Default for ImageNet but can customize this
        super().__init__() # don't forget the super().__init__()!

        # 3. Make the image size is divisble by the patch size
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        # 4. Calculate number of patches (height * width/patch^2)
        self.num_patches = (img_size * img_size) // patch_size**2

        # 5. Create learnable class embedding (needs to go at front of sequence of patch embeddings)
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        # 6. Create learnable position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)

        # 7. Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # 8. Create patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        # 9. Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential())
        # Note: The "*" means "all"
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])

        # 10. Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    # 11. Create a forward() method
    def forward(self, x):

        # 12. Get batch size
        batch_size = x.shape[0]

        # 13. Create class token embedding and expand it to match the batch size (equation 1)
        class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)

        # 14. Create patch embedding (equation 1)
        x = self.patch_embedding(x)

        # 15. Concat class embedding and patch embedding (equation 1)
        x = torch.cat((class_token, x), dim=1)

        # 16. Add position embedding to patch embedding (equation 1)
        x = self.position_embedding + x

        # 17. Run embedding dropout (Appendix B.1)
        x = self.embedding_dropout(x)

        # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(x)

        # 19. Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x