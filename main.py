from models import (
    transformer as t)
import argparse
from train import train_bert, train_gpt
from config import BertConfig, GPTConfig
from utils import create_bert_dataset, create_gpt_dataset




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type=str, help='model type')
    args = parser.parse_args()
    # make user input case insensitive
    args.model_type = args.model_type.lower()
    
    if args.model_type not in ['bert', 'gpt']:
        raise ValueError("Mode must be 'GPT' or 'BERT'")
    
    if args.model_type == 'bert':
        config = BertConfig()
        config.n_epochs = 100
        dataset, data_loader, vocab = create_bert_dataset(config)
        model = t.BERT(config.n_code,
                    config.n_heads,
                    config.embed_size,
                    config.inner_ff_size,
                    len(dataset.vocab),
                    config.seq_len,
                    config.dropout)
        train_bert(model, dataset, data_loader, vocab, config)
        
    if args.model_type == 'gpt':
        config = GPTConfig()
        config.n_epochs = 100
        train_data, val_data, tokenizer, vocab_size = create_gpt_dataset(config)
        model = t.GPT(config.vocab_size,
                    config.num_embed,
                    config.block_size,
                    config.num_heads,
                    config.num_layers,
                    config.dropout,
                    config.device)
        train_gpt(model, train_data, val_data, config)
    
    
    