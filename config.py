from pathlib import Path
import torch
from torch import nn
import os
from torchvision import transforms
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
    
    
# ===================== ViT Config ===================== #

class VITConfig:
    def __init__(self):
        self._patch_size = 16
        self._img_size = 224
        self._img_path = 'dataset/pizza_steak_sushi'
        self.batch_size = 32
        self.num_workers = os.cpu_count()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = 0.2
        
    @property
    def patch_size(self):
        return self._patch_size
    
    @property
    def img_size(self):
        return self._img_size
    
    @img_size.setter
    def img_size(self, value):
        self._img_size = value
    
    @patch_size.setter
    def patch_size(self, value):
        self._patch_size = value
        
    @property
    def num_patches(self):
        return self._img_size/self._patch_size
        
    @property
    def conv2d(self):
        return nn.Conv2d(in_channels=3, # number of color channels
                    out_channels=768, # from Table 1: Hidden size D, this is the embedding size
                    kernel_size=self._patch_size, # could also use (patch_size, patch_size)
                    stride=self._patch_size,
                    padding=0)
        
    @property
    def manual_transform(self):
        return transforms.Compose([
                            transforms.Resize((self._img_size,self._img_size)),
                            transforms.ToTensor()])
        
    @property
    def train_dir(self):
        return self._img_path + '/train'
    
    @property
    def test_dir(self):
        return self._img_path + '/test'
        
    