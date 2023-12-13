#%%
import csv
import random
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

import os

def get_sequences_from_csv(csv_file_path):
    sequences = []
    with open(csv_file_path, "r") as f:
        reader = csv.reader(f)
        for i, data in enumerate(reader):
            if i % 2 == 0:
                sequence = data[1:]
                sequences.append(sequence)
    return sequences

def get_sequences_from_directory(directory_path):
    all_sequences = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            sequences = get_sequences_from_csv(file_path)
            all_sequences.extend(sequences)
    return all_sequences

def struct_dataset(sequences):
    data = []
    for line in sequences:
        seq = []
        for node in line:
            if len(seq) == 0:
                seq.append(node)
                continue
            if node != seq[-1]:
                seq.append(node)
        data.append(seq)
    return data

def filter_dataset(sequences, target_len=5):
    data = []
    for line in sequences:
        if len(line) >= target_len:
            #start_idx = random.randint(0, len(line) - target_len)
            #data.append(line[start_idx:start_idx + target_len])
            data.append(line)
    return data

def get_total_nodes(sequences):
    nodes = set()
    for line in sequences:
        for node in line:
            nodes.add(node)
    return list(nodes)

def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max

class PangyoDataset(Dataset):
    def __init__(self, data, node_info, seq_len=8):
        self.data = data
        self.node_info = node_info
        self.seq_len = seq_len
        self.nodes2idx = self.node_info["node2idx"]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        line = [self.nodes2idx[node] for node in line]
        
        # 랜덤하게 시작 인덱스 선택
        start_idx = random.randint(0, len(line) - self.seq_len)
        line = line[start_idx:start_idx + self.seq_len]
        #line = line[:self.seq_len]
        
        session = np.array(line[:-1])
        session = torch.from_numpy(session).long()
        label = np.array(line[-1])
        label = torch.from_numpy(label).long()
        
        inputs = torch.zeros(self.seq_len).long()
        inputs[:len(session)] = session
        mask = torch.zeros(self.seq_len).bool()
        mask[:len(session)] = True
        return inputs, label, mask

def collate_fn(batch):
    inputs, targets, mask = zip(*batch)
    inputs = torch.stack(inputs)
    mask = torch.stack(mask)
    targets = torch.stack(targets)
   
    items, n_node, A, alias_inputs = [], [], [], []
    # Items : List of items in each session with ascending order
    # n_node : Number of unique items in each session with zero
    # A : Adjacency matrix of each session
    # alias_inputs : List of indices of items in each session
    for u_input in inputs:
        n_node.append(len(np.unique(u_input)))
    max_n_node = max(n_node) # Most number of nodes in a graph

    for u_input in inputs:
        u_input = u_input.numpy()
        node = np.unique(u_input)
        items.append(node.tolist() + (max_n_node - len(node)) * [0])
        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            if u_input[i + 1] == 0:
                break
            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        A.append(u_A)
        alias_inputs.append([np.where(node == i)[0][0] for i in u_input])

    items = torch.Tensor(items).long()
    A = torch.Tensor(A).float()
    alias_inputs = torch.Tensor(alias_inputs).long()
    n_node = torch.Tensor(n_node).long()
    
    return alias_inputs, A, items, mask, targets, inputs

#시드 값 조정을 통해 다른 환경에서도 동일하게 shuffle된 데이터 사용
def set_seed(seed_value = 1209):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def get_total_nodes(sequences):
    nodes = set()
    for line in sequences:
        nodes.update(line)
    return list(nodes)
   

if __name__ == "__main__":
    # Setting the seed
    set_seed()
    directory_path = './datasets/00.processed_csv_file_ssp/15sequence/'
    all_sequences = get_sequences_from_directory(directory_path)
    # print(all_sequences)

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
    train_dataset = PangyoDataset(train_sequences, node_info, seq_len=15)
    valid_dataset = PangyoDataset(valid_sequences, node_info, seq_len=15)

    # DataLoader setup for training and validation
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=32)
    batch = next(iter(train_loader))
    print(batch)
# %%
