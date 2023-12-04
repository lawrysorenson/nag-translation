
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import random
import copy
import torch

def generate_mask(size, lens):
  ans = torch.ones((len(lens), size)) # TODO: dtype?
  for i, l in enumerate(lens):
    ans[i,l:] = 0
  return ans

def get_pad_to_longest(PAD):
    def pad_to_longest(batch):
        src, tgt_mask, tgt = zip(*batch)

        src_lens = [len(s) for s in src]
        pad_len = max(src_lens)
        src_mask = generate_mask(pad_len, src_lens)
        pad_src = [s + [PAD] * (pad_len - len(s)) for s in src]

        tgt_lens = [len(s) for s in tgt]
        tgt_mask_lens = [len(s) for s in tgt_mask]
        pad_len = max(max(tgt_lens), max(tgt_mask_lens))
        tgt_mask_mask = generate_mask(pad_len, tgt_mask_lens)
        pad_tgt_mask = [s + [PAD] * (pad_len - len(s)) for s in tgt_mask]
        pad_tgt = [s + [PAD] * (pad_len - len(s)) for s in tgt]

        pad_src = torch.tensor(pad_src)
        pad_tgt_mask = torch.tensor(pad_tgt_mask)
        pad_tgt = torch.tensor(pad_tgt)

        return pad_src, src_mask, pad_tgt_mask, tgt_mask_mask, pad_tgt

    return pad_to_longest


class TrainDataset(Dataset):
    def __init__(self, tokenizer = None, data = None, path = './data', train=False):
        self.train = train
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.convert_tokens_to_ids(['<mask>'])[0]
        if data is not None:
            self.data = data
        else:
            self.data = []
            src_path = os.path.join(path, 'src.txt')
            tgt_path = os.path.join(path, 'tgt.txt')
            with open(src_path, 'r') as src_in, open(tgt_path, 'r') as tgt_in:
                i = 0
                for src, tgt in zip(tqdm(src_in.readlines(), desc='Loading dataset'), tgt_in):
                    i += 1
                    if i == 200000: break # limit input for fast development
                    src=src.strip()
                    tgt=tgt.strip()

                    src = tokenizer.encode(src)
                    tgt = tokenizer.encode(tgt)

                    self.data.append((src, tgt))

    def train_test_split(self, train=0.8, val=0.1, test=0.1):
        if train + val + test != 1: raise ValueError('Train, val, and test must sum to 1')
        random.seed(0)
        random.shuffle(self.data)

        N = len(self.data)

        test_split = max(10000, int(N*test))
        val_split = max(10000, int(N*val)) + test_split

        train_dataset = self.data[:-val_split]
        val_dataset = self.data[-val_split:-test_split]
        test_dataset = self.data[-test_split:]

        train_dataset = TrainDataset(tokenizer=self.tokenizer, data=train_dataset, train=True)
        val_dataset = TrainDataset(tokenizer=self.tokenizer, data=val_dataset)
        test_dataset = TrainDataset(tokenizer=self.tokenizer, data=test_dataset)

        return train_dataset, val_dataset, test_dataset

    def set_train(self, train = True):
        self.train = train

    def __getitem__(self, i):
        src, tgt = self.data[i]
        tgt_mask = copy.copy(tgt)
        mask_inds = random.sample(range(len(tgt_mask)), int(len(tgt_mask) * 0.3))
        for i in mask_inds: tgt_mask[i] = self.mask_id
        return src, tgt_mask, tgt
  
    def __len__(self):
        return len(self.data)
