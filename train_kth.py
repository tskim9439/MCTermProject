#%%
import numpy as np
import pickle
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from utils import Data, split_validation
from kth_dataset import KTHDataset, collate_fn
from model import SessionGraph

cfg_fp = "config.yaml"
cfg = OmegaConf.load(cfg_fp)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv_file = r'/home/gtts/MCTermProject/datasets/2014_01_preprocess_with_time.csv'

#%%
"""
if cfg.dataset == 'diginetica':
    n_node = 43098 # number of items
else:
    NotImplementedError(f"Dataset {cfg.dataset} not implemented")
"""
# Import Dataset  
train_dataset = KTHDataset(csv_file)
#valid_dataset = DigineticaDataset(valid_data)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=cfg.batchSize,
                                           shuffle=True,
                                           collate_fn=collate_fn)
valid_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=cfg.batchSize,
                                           shuffle=False,
                                           collate_fn=collate_fn)

n_node = 1500 #len(train_dataset.nodes)
print("n_node: ", n_node)
"""
if cfg.dataset == 'diginetica':
    n_node = 43098 # number of items
else:
    NotImplementedError(f"Dataset {cfg.dataset} not implemented")
"""

# Import Model
model = SessionGraph(cfg, n_node+2).to(device)

# Train Model
# Run training loop
for epoch in range(cfg.epoch):
    tq = tqdm(train_loader, desc=f"Training [Epoch {epoch+1}/{cfg.epoch}]")
    model.train()
    for i, batch in enumerate(tq):
        alias_inputs, A, items, mask, targets, inputs = batch
        alias_inputs = alias_inputs.to(device)
        A = A.to(device)
        items = items.to(device)
        mask = mask.to(device)
        targets = targets.to(device)
        inputs = inputs.to(device)
        
        model.optimizer.zero_grad()
        # Forward
        hidden = model(items, A)
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        scores = model.compute_scores(seq_hidden, mask)

        loss = model.loss_function(scores, targets)
        
        loss.backward()
        model.optimizer.step()
        
        tq.set_postfix(loss=loss.item())

    # Run Validation
    tq = tqdm(valid_loader, desc=f"Validation [Epoch {epoch+1}/{cfg.epoch}]")
    model.eval()
    accuracy = []
    for i, batch in enumerate(tq):
        alias_inputs, A, items, mask, targets, inputs = batch
        alias_inputs = alias_inputs.to(device)
        A = A.to(device)
        items = items.to(device)
        mask = mask.to(device)
        targets = targets.to(device)
        inputs = inputs.to(device)
        
        # Forward
        hidden = model(items, A)
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        scores = model.compute_scores(seq_hidden, mask)
        
        scores = scores.detach().cpu().numpy()
        predictions = np.argmax(scores, axis=1)
        targets = targets.detach().cpu().numpy()
        acc = np.sum(predictions == targets) / len(targets)
        acc = acc * 100
        accuracy.append(acc)
        
        tq.set_postfix(acc=acc)

    print(f"Validation Results  [Epoch {epoch+1}/{cfg.epoch}]")
    print(f"Accuracy: {np.mean(accuracy)}")

# %%
