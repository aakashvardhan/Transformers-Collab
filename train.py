from torch.utils.data import Dataset
import torch
import numpy as np
import torch.nn.functional as F
from collections import Counter
from os.path import exists
import torch.optim as optim
import torch.nn as nn
import random
import math
import re
from utils import (get_batch_for_bert, 
                   save_embeddings, 
                   save_model,
                   get_batch_for_gpt,
                   estimate_loss,
                   save_model_to_checkpoint)
# import subprocess
# try:
#     import lightning as pl
#     from lightning.fabric import Fabric
# except:
#     print("Installing PyTorch Lightning...")
#     subprocess.run(["pip", "install", "lightning"])

# ========================= BERT =========================

def train_bert(model, dataset, data_loader, vocab, config):
    model = model.cuda()
    #optimizer
    optim_kwargs = {'lr':1e-4, 'weight_decay':1e-4, 'betas':(.9,.999)}
    print('initializing optimizer and loss...')
    optimizer = optim.Adam(model.parameters(), **optim_kwargs)
    loss_model = nn.CrossEntropyLoss(ignore_index=dataset.IGNORE_IDX)   

    #3) training loop
    print('training...')
    print_each = 10
    model.train()
    batch_iter = iter(data_loader)
    for it in range(config.n_epochs):

        #get batch
        batch, batch_iter = get_batch_for_bert(data_loader, batch_iter)

        #infer
        masked_input = batch['input']
        masked_target = batch['target']

        masked_input = masked_input.cuda(non_blocking=True)
        masked_target = masked_target.cuda(non_blocking=True)
        output = model(masked_input)

        #compute the cross entropy loss
        output_v = output.view(-1,output.shape[-1])
        target_v = masked_target.view(-1,1).squeeze()
        loss = loss_model(output_v, target_v)

        #compute gradients
        loss.backward()

        #apply gradients
        optimizer.step()

        #print step
        if it % print_each == 0:
            print('it:', it,
                ' | loss', np.round(loss.item(),2),
                ' | Δw:', round(model.embeddings.weight.grad.abs().sum().item(),3))

        #reset gradients
        optimizer.zero_grad()

    # ========================= SAVE =========================
    #save model
    save_model(model)

    #save embeddings
    save_embeddings(dataset, model)
    

# ========================= GPT =========================

def train_gpt(model,train_data, val_data, config):
    # optimizer takes the model's parameters and the learning rate as input,
    # and updates the parameters during the training process in order to
    # minimize the loss function.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    for step in range(config.max_iter):

        # every EVAL_INTER evaluate the loss on train and val sets
        if step % config.eval_inter == 0 or step == config.max_iter - 1:
            loss_train = estimate_loss(
                data=train_data, model=model, block_size=config.block_size, batch_size=config.batch_size, config=config
            )
            loss_val = estimate_loss(
                data=val_data, model=model, block_size=config.block_size, batch_size=config.batch_size, config=config
            )
            print("step {:10} | train loss {:6.4f} | val loss {:6.4f}".format(step, loss_train, loss_val))

        # sample a batch of data
        xb, yb = get_batch_for_gpt(data=train_data, block_size=config.block_size, batch_size=config.batch_size, config=config)
        logits, loss = model.forward(xb, yb)
        # zero_grad() method sets the gradients of all parameters in the optimizer to zero
        optimizer.zero_grad(set_to_none=True)
        # backward() method on the loss variable calculates the gradients
        # of the loss with respect to the model's parameters.
        loss.backward()
        # step() method on the optimizer updates the model's parameters
        # using the calculated gradients, in order to minimize the loss.
        optimizer.step()
        
    save_model_to_checkpoint(model=model, checkpoint_path="checkpoints", epoch=step)