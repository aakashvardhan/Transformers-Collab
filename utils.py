import re
import torch
import numpy as np
from collections import Counter
from os.path import exists
from dataset.bert_dataset import SentencesDataset
import os
from datetime import datetime
import subprocess
import zipfile
from pathlib import Path
import requests
import matplotlib.pyplot as plt
import random
from torch import nn
from models.vit_patch_embedding import PatchEmbedding

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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# Try to get torchinfo, if not, install it
try:
    from torchinfo import summary
except:
    print("Installing torchinfo...")
    subprocess.run(["pip", "install", "torchinfo"])
    from torchinfo import summary


# ========================= All Model Utilities =========================

def save_model(model: torch.nn.Module, model_name: str, target_dir: Path):
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)
    try:
        # Create model save path
        assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
        model_save_path = target_dir_path / model_name

        # Save the model state_dict()
        print(f"[INFO] Saving model to: {model_save_path}")
        torch.save(obj=model.state_dict(),
                f=model_save_path)
    except Exception as e:
        print(f"Could not save model to {model_save_path}")
        print(e)

# ========================= BERT =========================

def create_bert_dataset(config):
    # check if path exists, if not create it
    #1) load text
    print('loading text...')
    # sent_pth = 'dataset/training.txt'
    sentences = open(config.sent_pth).read().lower().split('\n')

    #2) tokenize sentences (can be done during training, you can also use spacy udpipe)
    print('tokenizing sentences...')
    special_chars = ',?;.:/*!+-()[]{}"\'&'
    sentences = [re.sub(f'[{re.escape(special_chars)}]', ' \g<0> ', s).split(' ') for s in sentences]
    sentences = [[w for w in s if len(w)] for s in sentences]

    #3) create vocab if not already created
    print('creating/loading vocab...')
    # vocab_pth = 'dataset/vocab.txt'
    if not exists(config.vocab_pth):
        words = [w for s in sentences for w in s]
        vocab = Counter(words).most_common(config.n_vocab) #keep the N most frequent words
        vocab = [w[0] for w in vocab]
        open(config.vocab_pth, 'w+').write('\n'.join(vocab))
    else:
        vocab = open(config.vocab_pth).read().split('\n')

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
    
    
    
# ========================= GPT =========================



def create_gpt_dataset(config):
    
    # raw data
    # path_do_data = "dataset/english.txt"
    data_raw = open(config.path_do_data, encoding="utf-8").read()
    # we use pretrained BERT tokenizer for performance improvements
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    # data_raw = data_raw[4000000:] # short dataset

    # train/val split
    data = encode(text_seq=data_raw, tokenizer=tokenizer)
    n = int(config.split_val * len(data))  # first 90% will be train, rest val
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
        
        
# ========================= ViT =========================



def create_dataloader(config: any):
    
    # Use the image folder function from torchvision.datasets to create datasets
    train_dataset = datasets.ImageFolder(config.train_dir, transform=config.manual_transform)
    test_dataset = datasets.ImageFolder(config.test_dir, transform=config.manual_transform)
    
    # get class names
    class_names = train_dataset.classes
    
    # Turn images into data loaders
    train_dataloader = DataLoader(train_dataset,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    num_workers=config.num_workers,
                                    pin_memory=True)
    test_dataloader = DataLoader(test_dataset,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=config.num_workers,
                                    pin_memory=True)
    return train_dataloader, test_dataloader, class_names

def get_img_batch(dataloader):
    # Get a batch of images
    image_batch, label_batch = next(iter(dataloader))

    # Get a single image from the batch
    image, label = image_batch[0], label_batch[0]
    
    return image, label

# Plot a single image and its label

def show_img(image, label, class_names):
    # Plot image with matplotlib
    plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
    plt.title(class_names[label])
    plt.axis(False)
    plt.show()
    # save image
    plt.savefig(f"image_{class_names[label]}.png")
    

def patchify_img(image, label, class_names, config):
    image_permuted = image.permute(1, 2, 0)
    patch_size = config.patch_size
    img_size = config.img_size
    num_patches = img_size / patch_size
    assert img_size % patch_size == 0, "Image size must be divisible by patch size"
    print(f"Number of patches per row: {num_patches}\
            \nNumber of patches per column: {num_patches}\
            \nTotal patches: {num_patches*num_patches}\
            \nPatch size: {patch_size} pixels x {patch_size} pixels")

    # Create a series of subplots
    fig, axs = plt.subplots(nrows=img_size // patch_size, # need int not float
                            ncols=img_size // patch_size,
                            figsize=(num_patches, num_patches),
                            sharex=True,
                            sharey=True)

    # Loop through height and width of image
    for i, patch_height in enumerate(range(0, img_size, patch_size)): # iterate through height
        for j, patch_width in enumerate(range(0, img_size, patch_size)): # iterate through width

            # Plot the permuted image patch (image_permuted -> (Height, Width, Color Channels))
            axs[i, j].imshow(image_permuted[patch_height:patch_height+patch_size, # iterate through height
                                            patch_width:patch_width+patch_size, # iterate through width
                                            :]) # get all color channels

            # Set up label information, remove the ticks for clarity and set labels to outside
            axs[i, j].set_ylabel(i+1,
                                rotation="horizontal",
                                horizontalalignment="right",
                                verticalalignment="center")
            axs[i, j].set_xlabel(j+1)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].label_outer()

    # Set a super title
    fig.suptitle(f"{class_names[label]} -> Patchified", fontsize=16)
    plt.show()
    # save figure
    fig.savefig(f"image_{class_names[label]}_patchified.png")
    
def show_conv2d_feature_maps(image, config, k=5):
    image_out_of_conv = config.conv2d(image.unsqueeze(0)) # add extra dimension for batch size
    random_indexes = random.sample(range(0, 758), k=k) # pick 5 numbers between 0 and the embedding size
    print(f"Showing random convolutional feature maps from indexes: {random_indexes}")

    # Create plot
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(12, 12))

    # Plot random image feature maps
    for i, idx in enumerate(random_indexes):
        image_conv_feature_map = image_out_of_conv[:, idx, :, :] # index on the output tensor of the convolutional layer
        axs[i].imshow(image_conv_feature_map.squeeze().detach().numpy())
        axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[]);
        # save figure
        plt.savefig(f"image_conv_feature_map_{i}.png")
        
def show_flattened_feature_map(image, config):
    # Create flatten layer
    flatten = nn.Flatten(start_dim=2, # flatten feature_map_height (dimension 2)
                        end_dim=3) # flatten feature_map_width (dimension 3)
    # 2. Turn image into feature maps
    image_out_of_conv = config.conv2d(image.unsqueeze(0)) # add batch dimension to avoid shape errors
    print(f"Image feature map shape: {image_out_of_conv.shape}")

    # 3. Flatten the feature maps
    image_out_of_conv_flattened = flatten(image_out_of_conv)
    print(f"Flattened image feature map shape: {image_out_of_conv_flattened.shape}")
    
    # Get flattened image patch embeddings in right shape
    image_out_of_conv_flattened_reshaped = image_out_of_conv_flattened.permute(0, 2, 1) # [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
    print(f"Patch embedding sequence shape: {image_out_of_conv_flattened_reshaped.shape} -> [batch_size, num_patches, embedding_size]")
    
    # Get a single flattened feature map
    single_flattened_feature_map = image_out_of_conv_flattened_reshaped[:, :, 0] # index: (batch_size, number_of_patches, embedding_dimension)

    # Plot the flattened feature map visually
    plt.figure(figsize=(22, 22))
    plt.imshow(single_flattened_feature_map.detach().numpy())
    # save figure
    plt.savefig(f"image_flattened_feature_map.png")
    plt.title(f"Flattened feature map shape: {single_flattened_feature_map.shape}")
    plt.axis(False);
    
# get vit model summary

def get_vit_model_summary(model, input_size=(32, 3, 224, 224), col_names=["input_size", 
                                                                         "output_size", 
                                                                         "num_params", 
                                                                         "trainable"]):
    print(summary(model,
                   input_size=input_size,
                   col_names=col_names,
                   col_width=20,
                   row_settings=["var_names"]))
    
def get_patch_pos_embedding(image, config):
    patch_size = config.patch_size
    # 2. Print shape of original image tensor and get the image dimensions
    print(f"Image tensor shape: {image.shape}")
    height, width = image.shape[1], image.shape[2]

    # 3. Get image tensor and add batch dimension
    x = image.unsqueeze(0)
    print(f"Input image with batch dimension shape: {x.shape}")

    # 4. Create patch embedding layer
    patch_embedding_layer = PatchEmbedding(in_channels=3,
                                        patch_size=patch_size,
                                        embedding_dim=768)

    # 5. Pass image through patch embedding layer
    patch_embedding = patch_embedding_layer(x)
    print(f"Patching embedding shape: {patch_embedding.shape}")

    # 6. Create class token embedding
    batch_size = patch_embedding.shape[0]
    embedding_dimension = patch_embedding.shape[-1]
    class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimension),
                            requires_grad=True) # make sure it's learnable
    print(f"Class token embedding shape: {class_token.shape}")

    # 7. Prepend class token embedding to patch embedding
    patch_embedding_class_token = torch.cat((class_token, patch_embedding), dim=1)
    print(f"Patch embedding with class token shape: {patch_embedding_class_token.shape}")

    # 8. Create position embedding
    number_of_patches = int((height * width) / patch_size**2)
    position_embedding = nn.Parameter(torch.ones(1, number_of_patches+1, embedding_dimension),
                                    requires_grad=True) # make sure it's learnable

    # 9. Add position embedding to patch embedding with class token
    patch_and_position_embedding = patch_embedding_class_token + position_embedding
    print(f"Patch and position embedding shape: {patch_and_position_embedding.shape}")
    
    return patch_and_position_embedding

def get_shape_class_token(config):
    batch_size = config.batch_size
    class_token_embedding_single = nn.Parameter(data=torch.randn(1, 1, 768)) # create a single learnable class token
    class_token_embedding_expanded = class_token_embedding_single.expand(batch_size, -1, -1) # expand the single learnable class token across the batch dimension, "-1" means to "infer the dimension"

    # Print out the change in shapes
    print(f"Shape of class token embedding single: {class_token_embedding_single.shape}")
    print(f"Shape of class token embedding expanded: {class_token_embedding_expanded.shape}")
    