import os
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel 
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch import amp
from Our_HAR_read_data import * 
from model import *
from dataset import *

model_name = "bert-base-uncased"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
source = "uci"
target = "motion"
model_save_path = ""
beta = 0.5




if __name__ == "__main__":
    source_text, source_data, \
    source_text2, source_data2, \
    source_text3, source_data3, \
    target_text, target_data, \
    uci_text, shoaib_text = read_data(source, target)
    
    data1 = generate_step1(source_text, source_data, source)
    data2 = generate_step2(target_text, target_data, source_text, source_data, source_text2, source_data2, source_text3, source_data3)
    data3 = generate_step3(target_text, target_data)
    
    
    dataset1 = AllPairsDatasetContrastive(data1, tokenizer)
    dataset2 = AllPairsDataset(data2, tokenizer)
    dataset3 = AllPairsDataset(data3, tokenizer)
    
    collate_simple, collate_contrastive = make_collate_fn(tokenizer, pad_time_series=False)
    
    dataloader1 = DataLoader(dataset1, batch_size=16, shuffle=True, collate_fn=collate_contrastive, num_workers=0)
    dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=True, collate_fn=collate_simple, num_workers=0)
    dataloader3 = DataLoader(dataset3, batch_size=32, shuffle=False, collate_fn=collate_simple, num_workers=0)
    
    
    
    
    model = Text_SimilarityModel(bert_model=model_name, max_len=512, stride=128, pool="mean").float().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    triplet_loss_fn = TripletLoss(margin=1.0)
    
    bert_params = model.module.bert.parameters() if isinstance(model, nn.DataParallel) else model.bert.parameters()
    optimizer = AdamW(bert_params, lr=1e-5)
    
    num_epochs = 100
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
    
        for (time_series,
             input_ids1, attention_mask1,
             input_ids2, attention_mask2,
             labels,
             input_ids3, attention_mask3,
             input_ids4, attention_mask4,
             input_ids5, attention_mask5,
             input_ids6, attention_mask6) in dataloader1:
    
            optimizer.zero_grad()
            time_series = time_series.to(device).float()
    
            label_text = generate_label_new(source_text, source)
            label_emb_chunks = []
            with torch.no_grad():  
                for txt in label_text:
                    e = tokenizer(txt, add_special_tokens=True, truncation=False, return_tensors="pt")
                    e = {k: v.to(device) for k, v in e.items()}
                    if isinstance(model, nn.DataParallel):
                        emb = model.module._bert_embed_long(e["input_ids"], e["attention_mask"])
                    else:
                        emb = model._bert_embed_long(e["input_ids"], e["attention_mask"])
                    label_emb_chunks.append(emb)  # (1,H)
            label_emb = torch.cat(label_emb_chunks, dim=0)  # (C,H)
    
            outputs = model(
                input_ids1.to(device), attention_mask1.to(device),
                input_ids2.to(device), attention_mask2.to(device),
                time_series,
                input_ids3.to(device), attention_mask3.to(device),
                input_ids4.to(device), attention_mask4.to(device),
                input_ids5.to(device), attention_mask5.to(device),
                input_ids6.to(device), attention_mask6.to(device),
                labels.to(device)
            )
            (similarity_matrix1, similarity_matrix2,
             embeddings1, embeddings2,
             sensor_embeddings, sensor_embeddings2,
             anchor_embeddings1_1, positive_embeddings2_1, negative_embeddings3_1,
             anchor_embeddings1_2, positive_embeddings2_2, negative_embeddings3_2,
             labels_dev) = outputs
    
    
            logits_cls = torch.matmul(embeddings1, label_emb.T)  # (B,C)
            loss1 = custom_loss1(logits_cls, labels_dev.to(device))
            loss2 = triplet_loss_fn(anchor_embeddings1_1, positive_embeddings2_1, negative_embeddings3_1)
            loss3 = triplet_loss_fn(anchor_embeddings1_2, positive_embeddings2_2, negative_embeddings3_2)
            loss = loss1 + loss2 + loss3
            total_loss += loss.item()
    
    
            loss.backward()
            optimizer.step()
    
    
            with torch.no_grad():
                preds = torch.argmax(logits_cls, dim=1)  # (B,)
                correct_predictions += (preds == labels_dev.to(device)).sum().item()
                total_predictions += labels_dev.numel()
    
        avg_loss = total_loss / max(1, len(dataloader1))
        accuracy = correct_predictions / max(1, total_predictions)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")
    
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            path = os.path.join(model_save_path, f"{source}_best_model.pth")
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, path)
            print(f"New best model saved to: {path} (acc={accuracy:.4f})")

    