from pathlib import Path
import torch

# ===================== BERT Config ===================== #

class BertConfig:
    def __init__(self):
        self._embed_size = 128
        self.batch_size = 1024
        self.seq_len = 20
        self.n_epochs = 10000
        self.n_heads = 8
        self.n_code = 8
        self.n_vocab = 40000
        self.dropout = 0.1
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.betas = (.9, .999)
        self.sent_pth = 'dataset/training.txt'
        self.vocab_pth = 'dataset/vocab.txt'
        self.train_file = "dataset/training.txt"
        self.vocab_file = "dataset/vocab.txt"

    @property
    def embed_size(self):
        return self._embed_size

    @embed_size.setter
    def embed_size(self, value):
        self._embed_size = value

    @property
    def inner_ff_size(self):
        return self._embed_size * 4

# ===================== GPT Config ===================== #


class GPTConfig:
    def __init__(self):
        self.batch_size = 128
        self.path_do_data = "dataset/english.txt"
        self.block_size = 64
        self.split_val = 0.9
        self.max_iter = 500
        self.eval_inter = 100
        self.lr = 3e-4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._num_head = 6
        self.num_layer = 6
        self.dropout = 0.2
        
    @property
    def num_head(self):
        return self._num_head
    
    @num_head.setter
    def num_head(self, value):
        self._num_head = value
        
    @property
    def num_embed(self):
        return self._num_head * 128