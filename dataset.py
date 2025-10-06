from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class AllPairsDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        time_series, s1, s2, label = self.data[idx]
        e1 = self.tokenizer(s1, add_special_tokens=True, truncation=False)
        e2 = self.tokenizer(s2, add_special_tokens=True, truncation=False)

        input_ids1 = torch.tensor(e1["input_ids"], dtype=torch.long)
        attn_mask1 = torch.tensor(e1["attention_mask"], dtype=torch.long)
        input_ids2 = torch.tensor(e2["input_ids"], dtype=torch.long)
        attn_mask2 = torch.tensor(e2["attention_mask"], dtype=torch.long)

        return (
            torch.as_tensor(time_series).float(),  
            input_ids1, attn_mask1,
            input_ids2, attn_mask2,
            torch.tensor(label, dtype=torch.long)
        )

class AllPairsDatasetContrastive(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        time_series, s1, s2, label, s3, s4, s5, s6 = self.data[idx]

        def enc(text):
            e = self.tokenizer(text, add_special_tokens=True, truncation=False)
            return (torch.tensor(e["input_ids"], dtype=torch.long),
                    torch.tensor(e["attention_mask"], dtype=torch.long))

        input_ids1, attn_mask1 = enc(s1)
        input_ids2, attn_mask2 = enc(s2)
        input_ids3, attn_mask3 = enc(s3)
        input_ids4, attn_mask4 = enc(s4)
        input_ids5, attn_mask5 = enc(s5)
        input_ids6, attn_mask6 = enc(s6)

        return (
            torch.as_tensor(time_series).float(),
            input_ids1, attn_mask1,
            input_ids2, attn_mask2,
            torch.tensor(label, dtype=torch.long),
            input_ids3, attn_mask3,
            input_ids4, attn_mask4,
            input_ids5, attn_mask5,
            input_ids6, attn_mask6
        )

# -------------------------
# collate_fn：批次内动态 padding
# -------------------------
from torch.nn.utils.rnn import pad_sequence

def make_collate_fn(tokenizer, pad_time_series=False):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def _pad_ids(list_ids):
        return pad_sequence(list_ids, batch_first=True, padding_value=pad_id)

    def _pad_mask(list_masks):
        return pad_sequence(list_masks, batch_first=True, padding_value=0)

    def collate_simple(batch):
        time_list, ids1, m1, ids2, m2, labels = [], [], [], [], [], []
        for item in batch:
            ts, i1, a1, i2, a2, y = item
            time_list.append(ts)
            ids1.append(i1); m1.append(a1)
            ids2.append(i2); m2.append(a2)
            labels.append(y)

        input_ids1 = _pad_ids(ids1)
        attn_mask1 = _pad_mask(m1)
        input_ids2 = _pad_ids(ids2)
        attn_mask2 = _pad_mask(m2)
        labels = torch.stack(labels, dim=0)

        if pad_time_series:
            time_series = pad_sequence(time_list, batch_first=True, padding_value=0.0)
        else:
            time_series = torch.stack(time_list, dim=0)

        return (time_series, input_ids1, attn_mask1, input_ids2, attn_mask2, labels)

    def collate_contrastive(batch):
        time_list = []
        ids1=[]; m1=[]
        ids2=[]; m2=[]
        ids3=[]; m3=[]
        ids4=[]; m4=[]
        ids5=[]; m5=[]
        ids6=[]; m6=[]
        labels=[]
        for item in batch:
            (ts,
             i1, a1,
             i2, a2,
             y,
             i3, a3,
             i4, a4,
             i5, a5,
             i6, a6) = item

            time_list.append(ts)
            ids1.append(i1); m1.append(a1)
            ids2.append(i2); m2.append(a2)
            ids3.append(i3); m3.append(a3)
            ids4.append(i4); m4.append(a4)
            ids5.append(i5); m5.append(a5)
            ids6.append(i6); m6.append(a6)
            labels.append(y)

        batch_time = pad_sequence(time_list, batch_first=True, padding_value=0.0) if pad_time_series \
                     else torch.stack(time_list, dim=0)

        return (
            batch_time,
            _pad_ids(ids1), _pad_mask(m1),
            _pad_ids(ids2), _pad_mask(m2),
            torch.stack(labels, dim=0),
            _pad_ids(ids3), _pad_mask(m3),
            _pad_ids(ids4), _pad_mask(m4),
            _pad_ids(ids5), _pad_mask(m5),
            _pad_ids(ids6), _pad_mask(m6)
        )

    return collate_simple, collate_contrastive
