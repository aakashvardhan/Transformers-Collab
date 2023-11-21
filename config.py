from pathlib import Path


# ===================== BERT Config ===================== #

class BertConfig:
    def __init__(self):
        self._embed_size = 128
        self.batch_size = 1024
        self.seq_len = 20
        self.n_heads = 8
        self.n_code = 8
        self.n_vocab = 40000
        self.dropout = 0.1
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.betas = (.9, .999)
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
    