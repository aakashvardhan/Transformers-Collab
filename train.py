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
from utils import get_batch, save_embeddings, save_model

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
    loader_iter = iter(data_loader)
    for it in range(config.n_epochs):

        #get batch
        batch, batch_iter = get_batch(data_loader, batch_iter)

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