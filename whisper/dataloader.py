import json
import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class LibriDataset(Dataset):
    def __init__(self, path, loadtarget=True, tokenizer=None, biasing=True):
        with open(path) as f:
            self.data = json.load(f)
        self.data_idx = list(self.data.keys())
        self.loadtarget = loadtarget
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        uttname = self.data_idx[index]
        data = self.data[uttname]
        data_path = data["fbank"]
        fbank = torch.load(data_path)
        target = data["words"].lower()
        if self.loadtarget and self.tokenizer is not None:
            target = self.tokenizer.encode(" "+target) + [self.tokenizer.tokenizer.eos_token_id]
        return uttname, fbank, target
    

def check_in_utt(tok_word, target):
    for i in range(len(target)):
        if target[i:i+len(tok_word)] == tok_word:
            return True
    return False


def make_lexical_tree(word_dict, subword_dict, word_unk):
    # node [dict(subword_id -> node), word_id, word_set[start-1, end]]
    root = [{}, -1, None]
    for w, wid in word_dict.items():
        if wid > 0 and wid != word_unk:  # skip <blank> and <unk>
            if True in [c not in subword_dict for c in w]:  # skip unknown subword
                continue
            succ = root[0]  # get successors from root node
            for i, c in enumerate(w):
                cid = subword_dict[c]
                if cid not in succ:  # if next node does not exist, make a new node
                    succ[cid] = [{}, -1, (wid - 1, wid)]
                else:
                    prev = succ[cid][2]
                    succ[cid][2] = (min(prev[0], wid - 1), max(prev[1], wid))
                if i == len(w) - 1:  # if word end, set word id
                    succ[cid][1] = wid
                succ = succ[cid][0]  # move to the child successors
    return root


def collate_wrapper(batch):
    uttnames = [i[0] for i in batch]
    fbank = torch.stack([i[1] for i in batch])
    tgt = [i[2] for i in batch]
    return uttnames, fbank, tgt


def get_dataloader(path, bs, shuffle=True, loadtarget=True, tokenizer=None, biasing=False):
    dataset = LibriDataset(path, loadtarget=loadtarget, tokenizer=tokenizer, biasing=biasing)
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        collate_fn=collate_wrapper,
        pin_memory=True,
    )