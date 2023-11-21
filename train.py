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
    #1) create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    #2) create loss function
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.IGNORE_IDX)

    #3) training loop
    print('training...')
    model.train()
    loader_iter = iter(data_loader)
    for i in range(config.n_epochs):
        #fetch batch
        batch, loader_iter = get_batch(data_loader, loader_iter)
        batch_input = batch['input'].to(config['device'])
        batch_target = batch['target'].to(config['device'])

        #forward pass
        optimizer.zero_grad()
        output = model(batch_input)

        #compute loss
        loss = criterion(output.view(-1, len(vocab)), batch_target.view(-1))
        loss.backward()
        optimizer.step()

        #print loss
        if i % 100 == 0:
            print(f'loss: {loss.item()}')

    # ========================= SAVE =========================
    #save model
    save_model(model)

    #save embeddings
    save_embeddings(dataset, model)