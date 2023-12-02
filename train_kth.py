#%%
import numpy as np
import pickle
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from utils import Data, split_validation
from data import KTHDataset, collate_fn
from model import SessionGraph

cfg_fp = "config.yaml"
cfg = OmegaConf.load(cfg_fp)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 6
csv_file = r'/home/gtts/MCTermProject/datasets/2014_01_preprocess_with_time.csv'

train_pkl = f"datasets/train_data_{seq_len}.pkl"
valid_pkl = f"datasets/valid_data_{seq_len}.pkl"
node_info_pkl = f"datasets/node_info_{seq_len}.pkl"

# Import Dataset
with open(train_pkl, "rb") as f:
        train_data = pickle.load(f)
    
with open(valid_pkl, "rb") as f:
    valid_data = pickle.load(f)

with open(node_info_pkl, "rb") as f:
    node_info = pickle.load(f)

train_dataset = KTHDataset(train_data, node_info, seq_len=seq_len)
valid_dataset = KTHDataset(valid_data, node_info, seq_len=seq_len)


train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=cfg.batchSize,
                                           shuffle=True,
                                           collate_fn=collate_fn)
valid_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=cfg.batchSize,
                                           shuffle=False,
                                           collate_fn=collate_fn)

n_node = len(node_info["total_node"])

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
