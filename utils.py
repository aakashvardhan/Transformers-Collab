import random
import math
import re
from torch.utils.data import Dataset
import torch
import numpy as np
from collections import Counter
from config import bert_config
from os.path import exists
from bert_dataset import SentencesDataset
import os


# ========================= BERT =========================

def create_bert_dataset(config):
    # check if path exists, if not create it
    #1) load text
    print('loading text...')
    sent_pth = 'dataset/training.txt'
    sentences = open(sent_pth).read().lower().split('\n')

    #2) tokenize sentences (can be done during training, you can also use spacy udpipe)
    print('tokenizing sentences...')
    special_chars = ',?;.:/*!+-()[]{}"\'&'
    sentences = [re.sub(f'[{re.escape(special_chars)}]', ' \g<0> ', s).split(' ') for s in sentences]
    sentences = [[w for w in s if len(w)] for s in sentences]

    #3) create vocab if not already created
    print('creating/loading vocab...')
    vocab_pth = 'dataset/vocab.txt'
    if not exists(vocab_pth):
        words = [w for s in sentences for w in s]
        vocab = Counter(words).most_common(config['n_vocab']) #keep the N most frequent words
        vocab = [w[0] for w in vocab]
        open(vocab_pth, 'w+').write('\n'.join(vocab))
    else:
        vocab = open(vocab_pth).read().split('\n')

    #4) create dataset
    print('creating dataset...')
    dataset = SentencesDataset(sentences, vocab, config['seq_len'])
    # kwargs = {'num_workers':n_workers, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}
    kwargs = {'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':config['batch_size']}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    
    return dataset, data_loader, vocab


def get_batch(loader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter

def save_embeddings(dataset, model,N=3000):
    print('saving embeddings...')
    np.savetxt('dataset/values.tsv', np.round(model.embeddings.weight.detach().cpu().numpy()[0:N], 2), delimiter='\t', fmt='%1.2f')
    s = [dataset.rvocab[i] for i in range(N)]
    open('dataset/names.tsv', 'w+').write('\n'.join(s) )


    print('end')
    
def save_model(model):
    # check if path exists, if not create it
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")
    print('saving model...')
    torch.save(model.state_dict(), 'saved_models/bert.pth')