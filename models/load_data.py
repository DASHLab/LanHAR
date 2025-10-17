import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, random_split

from dataset import *
from read_data import *

def load_data_stage1(source_name, tokenizer, batch_size = 10):
    
    source_text, source_data = read_data_stage1(source_name)
    
    data1 = generate_step1(source_text, source_data, source_name)
    dataset1 = AllPairsDatasetContrastive(data1, tokenizer)    
    collate_simple, collate_contrastive = make_collate_fn(tokenizer, pad_time_series=False)    
    dataloader1 = DataLoader(dataset1, batch_size, shuffle=True, collate_fn=collate_contrastive, num_workers=0)
    
    return dataloader1


def load_data_stage2(source_name, target_name, tokenizer, batch_size):
    source_text, source_data, target_text, target_data = read_data(source_name, target_name)
    
    data2 = generate_step2(target_text, target_data, source_text, source_data)
    data3 = generate_step3(target_text, target_data)
    

    dataset2 = AllPairsDataset(data2, tokenizer)
    dataset3 = AllPairsDataset(data3, tokenizer)
    
    collate_simple, collate_contrastive = make_collate_fn(tokenizer, pad_time_series=False)
    dataloader2 = DataLoader(dataset2, batch_size, shuffle=True, collate_fn=collate_simple, num_workers=0)
    dataloader3 = DataLoader(dataset3, batch_size, shuffle=False, collate_fn=collate_simple, num_workers=0)
    return dataloader2, dataloader3


def load_data_test(source_name, target_name, tokenizer, batch_size):
    source_text, source_data, target_text, target_data = read_data(source_name, target_name)
    
    data3 = generate_step3(target_text, target_data)
    dataset3 = AllPairsDataset(data3, tokenizer)
    
    collate_simple, collate_contrastive = make_collate_fn(tokenizer, pad_time_series=False)

    total_len = len(dataset3)
    valid_len = int(0.2 * total_len)
    test_len = total_len - valid_len
    
    valid_dataset, test_dataset = random_split(dataset3, [valid_len, test_len])
    
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_simple, num_workers=0)
    
    test_loader = DataLoader(dataset3, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_simple, num_workers=0)
    return valid_loader, test_loader

