import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


class DigineticaDataset(Dataset):
    """
    Diginetica Dataset
    Args :
        data : tuple of (seqs of inputs [List[List]], labels [List])
    """
    def __init__(self, data, graph=None):
        self.sessions = data[0]
        self.labels = data[1]
        self.graph = graph
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