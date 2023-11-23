#%%
import csv

with open("train-item-views.csv", "r") as f:
    reader = csv.DictReader(f, delimiter=';')

    for i, data in enumerate(reader):
        print(data)
        if i == 3:
            break
# %%
import pickle

with open("/home/gtts/SR-GNN/datasets/diginetica/train.txt", "rb") as f:
    test = pickle.load(f)
test
# %%
test[1]
# %%
