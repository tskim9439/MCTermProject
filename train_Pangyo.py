import os
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import AverageMeter
from data_pangyo import PangyoDataset_Train, PangyoDataset_Valid, get_sequences_from_directory, collate_fn, get_total_nodes
from models import get_model

cfg_fp = "config.yaml"
cfg = OmegaConf.load(cfg_fp)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 13

# Logging
# Configuration and initialization
log_dir = f"./logs/{cfg.model.name}_{seq_len}"
os.makedirs(log_dir, exist_ok=True)

# Set the seed and load data
directory_path = f'C:/Users/HONGGU/.conda/MCterm/datasets/00.processed_csv_file_ssp/{seq_len}sequence/'
random.seed(100)
all_sequences = get_sequences_from_directory(directory_path)
random.shuffle(all_sequences)

# Create node_info and calculate n_node
total_nodes = get_total_nodes(all_sequences)
node2idx = {node: idx for idx, node in enumerate(total_nodes)}
idx2node = {idx: node for node, idx in node2idx.items()}
node_info = {
    "node2idx": node2idx,
    "idx2node": idx2node,
    "total_node": total_nodes
}
n_node = len(total_nodes)  # The number of unique nodes

# Split and load datasets
split_point = int(len(all_sequences) * 0.8)
train_sequences = all_sequences[:split_point]
valid_sequences = all_sequences[split_point:]

train_dataset = PangyoDataset_Train(train_sequences, node_info)
valid_dataset = PangyoDataset_Valid(valid_sequences, node_info)
print(valid_dataset)

train_loader = DataLoader(train_dataset, batch_size=cfg.batchSize, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=cfg.batchSize, shuffle=False, collate_fn=collate_fn)

# Import Model
model = get_model(cfg=cfg, num_classes=n_node + 2).to(device)

# Train Model
# Run training loop
logs = ["Epoch,Loss,ACC,HITS,HRR\n"]
val_accs = []
train_losses = []
best_acc = 0
for epoch in range(cfg.epoch):
    train_loss_meter = AverageMeter()
    
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
        if cfg.model.name == "srgnn":
            hidden = model(items, A)
            get = lambda i: hidden[i][alias_inputs[i]]
            seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
            scores = model.compute_scores(seq_hidden, mask)
        elif cfg.model.name in ["lstm", "gru", "transformer"]:
            scores = model(inputs)
        else:
            raise NotImplementedError

        loss = model.loss_function(scores, targets)
        
        loss.backward()
        model.optimizer.step()
        
        train_loss_meter.update(loss.item(), inputs.size(0))
        
        tq.set_postfix(loss=train_loss_meter.avg)
    train_losses.append(train_loss_meter.avg)

    # Run Validation
    tq = tqdm(valid_loader, desc=f"Validation [Epoch {epoch+1}/{cfg.epoch}]")
    model.eval()
    accuracy = []
    hits, mrrs = [], []
    
    for i, batch in enumerate(tq):
        alias_inputs, A, items, mask, targets, inputs = batch
        alias_inputs = alias_inputs.to(device)
        A = A.to(device)
        items = items.to(device)
        mask = mask.to(device)
        targets = targets.to(device)
        inputs = inputs.to(device)
        # print(items[0:])

        # Forward
        if cfg.model.name == "srgnn":
            hidden = model(items, A)
            get = lambda i: hidden[i][alias_inputs[i]]
            seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
            scores = model.compute_scores(seq_hidden, mask)
        elif cfg.model.name in ["lstm", "gru", "transformer"]:
            scores = model(inputs)
        else:
            raise NotImplementedError
        
        # Calculate Accuracies
        scores = scores.detach().cpu().numpy()
        predictions = np.argmax(scores, axis=1)
        targets = targets.detach().cpu().numpy()
        acc = np.sum(predictions == targets) / len(targets)
        acc = acc * 100
        accuracy.append(acc)
        tq.set_postfix(acc=acc)
        
    print(f"Validation Results  [Epoch {epoch+1}/{cfg.epoch}]")
    print(f"Accuracy: {np.mean(accuracy)}")
    
    log_line = f"{epoch+1},{np.mean(accuracy)}\n"
    logs.append(log_line)
    val_accs.append(np.mean(accuracy))

    if np.mean(accuracy) > best_acc:
        best_acc = np.mean(accuracy)
        torch.save(model.state_dict(), f"{log_dir}/best_model.pth")
        print(f"Saved best model with accuracy {best_acc}")

# Save Logs
with open(f"{log_dir}/logs.txt", "w") as f:
    f.writelines(logs)

# Save Plot Accuracies
plt.plot(val_accs)
plt.title(f"{cfg.model.name} {seq_len} Validation Accuracies")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(f"{log_dir}/val_acc.png")

plt.plot(train_losses)
plt.title(f"{cfg.model.name} {seq_len} Train Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"{log_dir}/val_acc.png")
