#%%
import csv
import random
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split



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


class KTHDataset(Dataset):
    def __init__(self,
                 data,
                 node_info,
                 seq_len=10,):
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
def set_seed(seed_value=100):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def filter_valid_sequences(train_dataset, validation_dataset):
    # Train 데이터셋에서 모든 고유 AP 목록 생성
    train_aps = set()
    for data in train_dataset:
        train_aps.update(set(data))

    # valid 데이터셋 필터링 (train에 없는 ap 값이 있으면 해당 시퀀스는 검증에서 제외)
    filtered_valid_dataset = []
    for data in validation_dataset:
        if all(ap in train_aps for ap in data):
            filtered_valid_dataset.append(data)

    return filtered_valid_dataset


def make_dataset_pkl(csv_file, seq_len=10, valid_ratio=0.2):
    data = get_sequences_from_csv(csv_file)
    data = struct_dataset(data)
    data = filter_dataset(data, target_len=seq_len)          #시퀀스 길이 10 이상 필터링
    
    for i in range(100):
        train_data, valid_data = train_test_split(data, test_size=valid_ratio, random_state=100)
        
        train_nodes = get_total_nodes(train_data)
        valid_nodes = get_total_nodes(valid_data)
        print("Number of unique nodes in train dataset: ", len(train_nodes))
        print("Number of unique nodes in validation dataset: ", len(valid_nodes))
        
        not_included_node = []
        for node in valid_nodes:
            if node not in train_nodes:
                not_included_node.append(node)
                print(node)
        print("Number of nodes not included in train dataset: ", len(not_included_node))
        
        if len(not_included_node) == 0:
            print("All nodes in validation dataset are included in train dataset.")
            with open(f"train_data_{seq_len}.pkl", "wb") as f:
                pickle.dump(train_data, f)
            with open(f"valid_data_{seq_len}.pkl", "wb") as f:
                pickle.dump(valid_data, f)

            total_node = get_total_nodes(data)
            node2idx = {node: idx for idx, node in enumerate(total_node)}
            idx2node = {idx: node for idx, node in enumerate(total_node)}
            node_info = {}
            node_info["node2idx"] = node2idx
            node_info["idx2node"] = idx2node
            node_info["total_node"] = total_node
            
            with open(f"node_info_{seq_len}.pkl", "wb") as f:
                pickle.dump(node_info, f)
            
            break
        


if __name__ == "__main__":
    # Setting the seed
    set_seed()

    csv_file = r'/home/gtts/MCTermProject/datasets/2014_01_preprocess_with_time.csv'
    make_dataset_pkl(csv_file, seq_len=6, valid_ratio=0.2)
    
    train_pkl = "datasets/train_data_10.pkl"
    valid_pkl = "datasets/valid_data_10.pkl"
    node_info_pkl = "datasets/node_info_10.pkl"
    
    with open(train_pkl, "rb") as f:
        train_data = pickle.load(f)
    
    with open(valid_pkl, "rb") as f:
        valid_data = pickle.load(f)
    
    with open(node_info_pkl, "rb") as f:
        node_info = pickle.load(f)
    
    train_dataset = KTHDataset(train_data, node_info)
    valid_dataset = KTHDataset(valid_data, node_info)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(train_loader))
    print(batch)
    
# %%
