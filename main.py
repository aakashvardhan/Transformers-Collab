from models import (
    transformer as t)
import argparse
from utils.gpt_utils import DEVICE
from train import train_bert
from config import bert_config
from utils import create_bert_dataset
# i want to train the bert model and use arg parser to get the parameters
# i want to use the bert model to train the model



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type=str, help='model type')
    
    if parser == 'bert':
        config = bert_config()
        dataset, data_loader, vocab = create_bert_dataset(config)
        model = t.BERT(config['n_code'],
                    config['n_heads'],
                    config['embed_size'],
                    config['inner_ff_size'],
                    len(dataset.vocab),
                    config['seq_len'],
                    config['dropout'])
        train_bert(model, dataset, data_loader, vocab, config)
    
    
    