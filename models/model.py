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
from time_transformer import * 
t = 1.0
model_name = "bert-base-uncased"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return nn.functional.normalize(x, p=2, dim=1)

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)

class Text_SimilarityModel(nn.Module):
    def __init__(self, bert_model="bert-base-uncased", max_len=512, stride=128, pool="mean"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.hidden_size = self.bert.config.hidden_size
        self.max_len = max_len
        self.stride = stride
        self.pool = pool

        self.sensor_encoder = TimeSeriesTransformer(
            input_dim=6, d_model=self.hidden_size, nhead=4,
            num_encoder_layers=6, dim_feedforward=2048, dropout=0.1
        )
        self.sensor_encoder2 = TimeSeriesTransformer(
            input_dim=6, d_model=self.hidden_size, nhead=4,
            num_encoder_layers=6, dim_feedforward=2048, dropout=0.1
        )
        self.normal = SimpleModel()

    def _pool_hidden(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        if self.pool == "cls":
            return last_hidden_state[:, 0, :]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B,L,1)
        summed = (last_hidden_state * mask).sum(dim=1)                  # (B,H)
        denom = mask.sum(dim=1).clamp(min=1e-6)                         # (B,1)
        return summed / denom

    def _bert_embed_long(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        B, L = input_ids.size()
        if L <= self.max_len:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return self._pool_hidden(out.last_hidden_state, attention_mask)  # (B,H)


        chunk_embs = []
        step = self.max_len - self.stride if self.max_len > self.stride else self.max_len
        device = input_ids.device
        for b in range(B):
            ids_b = input_ids[b]; mask_b = attention_mask[b]
            parts = []
            start = 0
            while start < L:
                end = min(start + self.max_len, L)
                ids_chunk = ids_b[start:end]
                mask_chunk = mask_b[start:end]

                # pad 到 max_len
                pad_len = self.max_len - (end - start)
                if pad_len > 0:
                    ids_chunk = torch.cat([ids_chunk, torch.full((pad_len,), tokenizer.pad_token_id or 0, dtype=ids_chunk.dtype, device=device)], dim=0)
                    mask_chunk = torch.cat([mask_chunk, torch.zeros(pad_len, dtype=mask_chunk.dtype, device=device)], dim=0)

                ids_chunk = ids_chunk.unsqueeze(0)
                mask_chunk = mask_chunk.unsqueeze(0)

                out = self.bert(input_ids=ids_chunk, attention_mask=mask_chunk)
                emb = self._pool_hidden(out.last_hidden_state, mask_chunk).squeeze(0)  # (H,)
                parts.append(emb)

                if end == L:
                    break
                start += step

            doc_emb = torch.stack(parts, dim=0).mean(dim=0)  # 简单平均聚合
            chunk_embs.append(doc_emb)
        return torch.stack(chunk_embs, dim=0)  # (B,H)

    def forward(self,
                input_ids1, attention_mask1,
                input_ids2, attention_mask2,
                time_series,
                input_ids3, attention_mask3,
                input_ids4, attention_mask4,
                input_ids5, attention_mask5,
                input_ids6, attention_mask6,
                labels):

        e1 = self._bert_embed_long(input_ids1, attention_mask1)  # (B,H)
        e2 = self._bert_embed_long(input_ids2, attention_mask2)
        e3 = self._bert_embed_long(input_ids3, attention_mask3)
        e4 = self._bert_embed_long(input_ids4, attention_mask4)
        e5 = self._bert_embed_long(input_ids5, attention_mask5)
        e6 = self._bert_embed_long(input_ids6, attention_mask6)

        anchor_embeddings1_1   = self.normal(e1)
        positive_embeddings2_1 = self.normal(e3)
        negative_embeddings3_1 = self.normal(e5)

        anchor_embeddings1_2   = self.normal(e2)
        positive_embeddings2_2 = self.normal(e4)
        negative_embeddings3_2 = self.normal(e6)


        sensor_embeddings  = self.sensor_encoder(time_series)
        sensor_embeddings2 = self.sensor_encoder2(time_series)


        similarity_matrix2 = torch.matmul(e1, e2.T)                         # (B,B)
        sim_left = sensor_embeddings.sum(dim=1)                              # (B,H)
        similarity_matrix1 = torch.matmul(sim_left, e1.T)                    # (B,B)

        return (similarity_matrix1, similarity_matrix2,
                e1, e2,
                sensor_embeddings, sensor_embeddings2,
                anchor_embeddings1_1, positive_embeddings2_1, negative_embeddings3_1,
                anchor_embeddings1_2, positive_embeddings2_2, negative_embeddings3_2,
                labels)



def symmetric_cross_entropy(logits, labels):
    loss_i = F.cross_entropy(logits, labels)
    logits_t = logits.transpose(0, 1)
    loss_t = F.cross_entropy(logits_t, labels)
    return (loss_i + loss_t) / 2

def symmetric_cross_entropy1(logits, labels):
    return F.cross_entropy(logits, labels)

def custom_loss(similarity_matrix, t):
    labels = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)
    return symmetric_cross_entropy(similarity_matrix, labels)

def custom_loss1(similarity_matrix, labels):
    labels = labels.to(similarity_matrix.device)
    return symmetric_cross_entropy1(similarity_matrix, labels)

def clip_loss(image_features, text_features, temperature=0.1):
    logits_per_image = (image_features @ text_features.T) / temperature
    logits_per_text = (text_features @ image_features.T) / temperature
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)
    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_text = F.cross_entropy(logits_per_text, labels)
    return (loss_img + loss_text) / 2
