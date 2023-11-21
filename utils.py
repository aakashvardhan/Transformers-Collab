import re
import torch
import numpy as np
from collections import Counter
from os.path import exists
from dataset.bert_dataset import SentencesDataset
import os
from datetime import datetime
import subprocess
try:
    from transformers import AutoTokenizer
except:
    print("Installing transformers pip...")
    subprocess.run(["pip", "install", "transformers"])
    from transformers import AutoTokenizer
    
from models.transformer import (
    GPT,
    BERT
)

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
        vocab = Counter(words).most_common(config.n_vocab) #keep the N most frequent words
        vocab = [w[0] for w in vocab]
        open(vocab_pth, 'w+').write('\n'.join(vocab))
    else:
        vocab = open(vocab_pth).read().split('\n')

    #4) create dataset
    print('creating dataset...')
    dataset = SentencesDataset(sentences, vocab, config.seq_len)
    # kwargs = {'num_workers':n_workers, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}
    kwargs = {'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':config.batch_size}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    
    return dataset, data_loader, vocab


def get_batch_for_bert(loader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter

def save_embeddings(dataset, model,N=3000):
    try:
        print('saving embeddings...')
        np.savetxt('dataset/values.tsv', np.round(model.embeddings.weight.detach().cpu().numpy()[0:N], 2), delimiter='\t', fmt='%1.2f')
    except Exception as e:
        print(f"Could not save embeddings to dataset/values.tsv")
        print(e)
    try:
        s = [dataset.rvocab[i] for i in range(N)]
        open('dataset/names.tsv', 'w+').write('\n'.join(s) )
        print('end')
    except Exception as e:
        print(f"Could not write to dataset/names.tsv")
        print(e)
    
def save_model(model):
    # check if path exists, if not create it
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")
    try:
        print('saving model...')
        torch.save(model.state_dict(), 'saved_models/bert.pth')
    except Exception as e:
        print(f"Could not save model to saved_models/bert.pth")
        print(e)
    
    
# ========================= GPT =========================



def create_gpt_dataset(split_val= 0.9):
    
    # raw data
    path_do_data = "dataset/english.txt"
    data_raw = open(path_do_data, encoding="utf-8").read()
    # we use pretrained BERT tokenizer for performance improvements
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    # data_raw = data_raw[4000000:] # short dataset

    # train/val split
    data = encode(text_seq=data_raw, tokenizer=tokenizer)
    n = int(split_val * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, tokenizer, vocab_size

def get_gpt_model(config: any, vocab_size: int):
    # train a new model
    model = GPT(
        vocab_size=vocab_size,
        num_embed=config.num_embed,
        block_size=config.block_size,
        num_heads=config.num_head,
        num_layers=config.num_layer,
        dropout=config.dropout,
    )
    # load model to GPU if available
    m = model.to(config.device)
    # print the number of parameters in the model
    print(
        "Model with {:.2f}M parameters".format(sum(p.numel() for p in m.parameters()) / 1e6)
    )
    return m


def encode(text_seq: str, tokenizer: any) -> torch.Tensor:
    """
    Function to encode input text using a pretrained tokenizer and vectorized lookups
    """
    # tokenize the input text
    tokens = tokenizer.tokenize(text_seq)
    # convert tokens to ids
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    # convert to torch tensor
    tokens_tensor = torch.tensor(tokens_ids, dtype=torch.long)
    return tokens_tensor

def decode(enc_sec: torch.Tensor, tokenizer: any) -> str:
    """
    Function to decode a tensor of token ids into a string
    """
    # convert the indices to a list of tokens
    enc_sec = enc_sec.tolist()
    # decode the indices to a string
    dec_sec = tokenizer.decode(enc_sec)
    return dec_sec

def get_batch_for_gpt(data: list[str], block_size:int, batch_size:int, config: any):
    """
    This is a simple function to create batches of data.
    GPUs allow for parallel processing we can feed multiple chunks of data at once.
    so thats why we need to create batches - how many independant sequences
    will be processed in parallel.
    
    Parameters:
    data: list[str]: data to take batches from
    block_size: int: size of the text that is processed at once
    batch_size: int: number of independant sequences to process in parallel
    
    Returns:
    x,y: a tuple of torch tensors with the input and target data
    """
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    # we stack batch_size row of sentences
    # so x and y are the matrices with rows_num = batch_size
    # and columns_num = block_size
    x = torch.stack([data[i:i+block_size] for i in ix])
    # y is x shifted by one since we are trying to predict the next token
    # word in y having all the previous words as context
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

@torch.no_grad()
def estimate_loss(
    data: list[str],
    model: torch.nn.Module,
    block_size: int,
    batch_size: int,
    eval_iters: int = 10,
    config: any = None,
):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch_for_gpt(data, block_size, batch_size, config)
        logits, loss = model(x, y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out

def load_model_from_checkpoint(
    model_class: torch.nn.Module,
    checkpoint_path: str = "checkpoints/state_dict_model.pt",
    **kwargs: dict,
) -> torch.nn.Module:
    
    try:
        state_dict = torch.load(checkpoint_path)
    except Exception as e:
        print(f"Could not load model from {checkpoint_path}")
        print(e)
        
    model = model_class(**kwargs)
    model.load_state_dict(state_dict)
    return model

def save_model_to_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str = "checkpoints/state_dict_model.pt", 
    epoch: int = 0):
    # check if path exists, if not create it
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
        
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY_H:M:S
    dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
    checkpoint_name = "checkpoint_epoch-" + str(epoch) + "_" + dt_string + ".pt"
    full_path = os.path.join(checkpoint_path, checkpoint_name)
    try:
        torch.save(model.state_dict(), full_path)
        print("Model saved to: {}".format(full_path))
        
    except Exception as e:
        print(f"Could not save model to {full_path}")
        print(e)