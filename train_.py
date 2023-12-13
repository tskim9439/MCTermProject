#%%
import os
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from tqdm import tqdm

from utils import Data, split_validation, AverageMeter
from data import KTHDataset, collate_fn
from models import get_model



cfg_fp = "config.yaml"
cfg = OmegaConf.load(cfg_fp)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = cfg.seq_len
csv_file = r'/home/gtts/MCTermProject/datasets/2014_01_preprocess_with_time.csv'

# Logging
log_dir = f"./logs/{cfg.model.name}_{seq_len}"
os.makedirs(log_dir, exist_ok=True)

#train_pkl = f"datasets/train_data_{seq_len}.pkl"
#valid_pkl = f"datasets/valid_data_{seq_len}.pkl"
#node_info_pkl = f"datasets/node_info_{seq_len}.pkl"
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

train_dataset = KTHDataset(train_data, node_info, seq_len=seq_len, split='train')
valid_dataset = KTHDataset(valid_data, node_info, seq_len=seq_len, split='valid')

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=cfg.batchSize,
                                           shuffle=True,
                                           collate_fn=collate_fn)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
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
        
        sub_scores = scores.topk(20)[1]
        sub_scores = sub_scores.detach().cpu().numpy()
        hit, mrr = [], []
        for score, target, m in zip(sub_scores, targets, mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
        
        hit = np.mean(hit) * 100
        mrr = np.mean(mrr) * 100
        
        
        # Calculate Accuracies
        scores = scores.detach().cpu().numpy()
        predictions = np.argmax(scores, axis=1)
        targets = targets.detach().cpu().numpy()
        acc = np.sum(predictions == targets) / len(targets)
        acc = acc * 100
        accuracy.append(acc)
        tq.set_postfix(acc=acc,
                       hit=hit,
                       mrr=mrr)

        hits.append(hit)
        mrrs.append(mrr)
    
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
