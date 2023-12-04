
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import random

class TrainDataset(Dataset):
    def __init__(self, tokenizer = None, data = None, path = './data', train=False):
        self.train = train
        if data is not None:
            self.data = data
        else:
            self.data = []
            src_path = os.path.join(path, 'src.txt')
            tgt_path = os.path.join(path, 'tgt.txt')
            with open(src_path, 'r') as src_in, open(tgt_path, 'r') as tgt_in:
                for src, tgt in zip(tqdm(src_in.readlines(), desc='Loading dataset'), tgt_in):
                    src=src.strip()
                    tgt=tgt.strip()

                    self.data.append((src, tgt))

    def train_test_split(self, train=0.8, val=0.1, test=0.1):
        if train + val + test != 1: raise ValueError('Train, val, and test must sum to 1')
        random.seed(0)
        random.shuffle(self.data)

        N = len(self.data)

        train_dataset = self.data[:int(N*train)]
        val_dataset = self.data[int(N*train):-int(N*test)]
        test_dataset = self.data[-int(N*test):]

        train_dataset = TrainDataset(data=train_dataset, train=True)
        val_dataset = TrainDataset(data=val_dataset)
        test_dataset = TrainDataset(data=test_dataset)

        return train_dataset, val_dataset, test_dataset

    def set_train(self, train = True):
        self.train = train

    def __getitem__(self, i):
        return self.data[i]
  
    def __len__(self):
        return len(self.data)
