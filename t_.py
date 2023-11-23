#%%
import csv
import random

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


def filter_dataset(sequences, target_len=6):
    data = []
    for line in sequences:
        if len(line) >= target_len:
            start_idx = random.randint(0, len(line) - target_len)
            data.append(line[start_idx:start_idx + target_len])
    return data

if __name__ == "__main__":
    data = get_sequences_from_csv("/home/gtts/MCTermProject/datasets/2014_01_preprocess_with_time.csv")
    data = struct_dataset(data)
    data = filter_dataset(data, target_len=6)

