from models import (
    transformer as t)
import argparse
from train import train_bert, train_gpt
from config import BertConfig, GPTConfig, VITConfig
from utils import (create_bert_dataset, 
                   create_gpt_dataset,
                   get_gpt_model,
                   create_dataloader)
from vit_engine import train, test
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type=str, help='model type')
    args = parser.parse_args()
    # make user input case insensitive
    args.model_type = args.model_type.lower()
    
    if args.model_type not in ['bert', 'gpt', 'vit']:
        raise ValueError("Mode must be 'GPT', 'BERT' or 'ViT'")
    
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
        model = get_gpt_model(config, vocab_size)
        train_gpt(model, train_data, val_data, config)
        
    if args.model_type == 'vit':
        config = VITConfig()
        train_dataloader, test_dataloader, class_names = create_dataloader(config)
        # Setup the optimizer to optimize our ViT model parameters using hyperparameters from the ViT paper
        
    
    
    