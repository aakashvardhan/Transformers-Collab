from models import (
    transformer as t)
import argparse
from train import train_bert, train_gpt
from config import BertConfig, GPTConfig, VITConfig
from utils import (create_bert_dataset, 
                   create_gpt_dataset,
                   get_gpt_model,
                   create_dataloader)
import torch.nn as nn
import torch
from pathlib import Path
import requests
import evaluation_utils as utils
import torchvision
import torchvision.transforms as transforms

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
        # 1. Get pretrained weights for ViT-Base
        class_names = ["pizza", "steak", "sushi"]
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT # requires torchvision >= 0.13, "DEFAULT" means best available

        # 2. Setup a ViT model instance with pretrained weights
        pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(config.device)
        # 3. Freeze the base parameters
        for parameter in pretrained_vit.parameters():
            parameter.requires_grad = False


        pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(config.device)
        pretrained_vit.load_state_dict(torch.load("models/08_pretrained_vit_feature_extractor_pizza_steak_sushi.pth"))
        with open("download.jpeg", "wb") as f:
        # When downloading from GitHub, need to use the "raw" file link
            request = requests.get("https://www.boss-pizza.co.uk/site/assets/images/uploads/2_3_5c232a9d83be_o.jpg")
            print(f"Downloading...")
            f.write(request.content)
            
        image_transform = transforms.Compose(
            [transforms.Resize((config.img_size, config.img_size)),
             transforms.ToTensor()])

        # Predict on custom image
        utils.pred_and_plot_image(model=pretrained_vit,
                            image_path="download.jpeg",
                            transform=image_transform,
                            class_names=class_names)
    
    