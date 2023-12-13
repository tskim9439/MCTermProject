#%%
import os
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from tqdm import tqdm

from utils import Data, split_validation, AverageMeter
from pangyo_dataset import PangyoDataset, collate_fn, get_sequences_from_directory, set_seed, get_total_nodes
from models import get_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


cfg_fp = "config.yaml"
cfg = OmegaConf.load(cfg_fp)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = cfg.seq_len
root_path = cfg.root_path
root_path = os.path.join(root_path, f"{seq_len}sequence")

# Logging
log_dir = f"./logs/{cfg.model.name}_{seq_len}"
os.makedirs(log_dir, exist_ok=True)

# Import Dataset
set_seed()
all_sequences = get_sequences_from_directory(root_path)

# Create node_info
total_nodes = get_total_nodes(all_sequences)
node2idx = {node: idx for idx, node in enumerate(total_nodes)}
idx2node = {idx: node for node, idx in node2idx.items()}
node_info = {
    "node2idx": node2idx,
    "idx2node": idx2node,
    "total_node": total_nodes
}

# Splitting data into training and validation sets
train_sequences, valid_sequences = train_test_split(all_sequences,
                                                train_size=0.8,
                                                test_size=0.2, random_state=100)

# Assuming you have a Dataset class to handle the sequences
# Replace 'YourDataset' with the name of your dataset class
train_dataset = PangyoDataset(train_sequences, node_info, seq_len=seq_len)
valid_dataset = PangyoDataset(valid_sequences, node_info, seq_len=seq_len)

train_loader = DataLoader(train_dataset,
                          batch_size=cfg.batchSize,
                          shuffle=True,
                          collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset,
                          batch_size=cfg.batchSize,
                          shuffle=False,
                          collate_fn=collate_fn)

n_node = len(node_info["total_node"])

# Import Model
model = get_model(cfg=cfg,
                  num_classes=n_node + 2).to(device)

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
        elif cfg.model.name in ["lstm", "gru"]:
            inputs = inputs[:, :-1] # Erase last timestamp (Padding)
            scores = model(inputs)
        else:
            raise NotImplementedError

        loss = model.loss_function(scores, targets)
        
        loss.backward()
        model.optimizer.step()
        
        train_loss_meter.update(loss.item(), inputs.size(0))
        
        tq.set_postfix(loss=train_loss_meter.avg,
                       lr=model.optimizer.param_groups[0]['lr'])
    
    train_losses.append(train_loss_meter.avg)
    model.scheduler.step()

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
        
        # Forward
        if cfg.model.name == "srgnn":
            hidden = model(items, A)
            get = lambda i: hidden[i][alias_inputs[i]]
            seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
            scores = model.compute_scores(seq_hidden, mask)
        elif cfg.model.name in ["lstm", "gru"]:
            inputs = inputs[:, :-1] # Erase last timestamp (Padding)
            scores = model(inputs)
        else:
            raise NotImplementedError
        
        # Calculate MRR
        scores = scores.detach().cpu()
        targets = targets.detach().cpu()
        
        
        # Calculate Accuracies
        scores = scores.detach().cpu().numpy()
        predictions = np.argmax(scores, axis=1)
        targets = targets.detach().cpu().numpy()
        acc = np.sum(predictions == targets) / len(targets)
        acc = acc * 100
        accuracy.append(acc)
        tq.set_postfix(acc=acc)
    
    if np.mean(accuracy) > best_acc:
        best_acc = np.mean(accuracy)
        torch.save(model.state_dict(), f"{log_dir}/best_model.pth")
        print(f"Saved best model with accuracy {best_acc}")
    
    print(f"Validation Results  [Epoch {epoch+1}/{cfg.epoch}]")
    print(f"Accuracy: {np.mean(accuracy)}")
    print(f"BEST Accuracy: {best_acc}")
    
    log_line = f"{epoch+1},{np.mean(accuracy)},{np.mean(hits)},{np.mean(mrrs)},{best_acc}\n"
    logs.append(log_line)
    val_accs.append(np.mean(accuracy))

# Save Logs
with open(f"{log_dir}/logs.txt", "w") as f:
    f.writelines(logs)

# Save Plot Accuracies
plt.plot(val_accs)
plt.title(f"{cfg.model.name} {seq_len} Validation Accuracies")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(f"{log_dir}/val_acc.png")
plt.close()

plt.plot(train_losses)
plt.title(f"{cfg.model.name} {seq_len} Train Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"{log_dir}/train_loss.png")
plt.close()

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of Model Parameters : {num_params}")
print(f"{cfg.model.ndame} {seq_len} Train Complete")
# %%
