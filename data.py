import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

def get_sequences_from_csv(csv_file_path):
    sequences = []
    with open(csv_file_path, "r") as f:
        reader = csv.reader(f)
        for i, data in enumerate(reader):
            if i % 2 == 0:
                sequence = data[1:]
                sequences.append(sequence)
    return sequences


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


def filter_dataset(sequences, target_len=10):
    data = []
    for line in sequences:
        if len(line) >= target_len:
            start_idx = random.randint(0, len(line) - target_len)
            data.append(line[start_idx:start_idx + target_len])
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


class KTHDataset(Dataset):
    def __init__(self, csv_file, graph=None):
        data = get_sequences_from_csv(csv_file)
        data = struct_dataset(data)
        data = filter_dataset(data, target_len=10)          #시퀀스 길이 10으로 고정
        self.nodes = get_total_nodes(data)
        self.nodes2idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.get_session_target(data)
    
    
    def get_session_target(self, data):
        sessions = []
        labels = []
        for line in data:
            session = line[:-1]
            label = line[-1]
            session = [self.nodes2idx[node] for node in session]
            label = self.nodes2idx[label]
            sessions.append(session)
            labels.append(label)
        
        self.sessions = sessions
        self.labels = labels
        self.len_max = max([len(s) for s in self.sessions])
    
    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        session = np.array(self.sessions[idx])
        session = torch.from_numpy(session).long()
        label = np.array(self.labels[idx])
        label = torch.from_numpy(label).long()
        
        inputs = torch.zeros(self.len_max).long()
        inputs[:len(session)] = session
        mask = torch.zeros(self.len_max).bool()
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

def set_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed()  # Setting the seed for reproducibility

if __name__ == "__main__":
    # Setting the seed
    set_seed()

    csv_file = r'datasets\\2014_01_preprocess_with_time.csv'
    dataset = KTHDataset(csv_file)

    # Splitting the dataset into train and validation sets
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    validation_size = total_size - train_size

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    # DataLoaders for train and validation sets with shuffling
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Example of iterating over the train_loader
    for alias_inputs, A, items, mask, targets, inputs in train_loader:
        print(alias_inputs)
        print(A)
        print(items)
        print(mask)
        print(targets)
        print(inputs)
        break
