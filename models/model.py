import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from sensor_encoder import *  

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

class lanhar(nn.Module):
    def __init__(self, bert_model="allenai/scibert_scivocab_uncased",
                 max_len=512, stride=128, pool="mean", pad_id=0):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.hidden_size = self.bert.config.hidden_size
        self.max_len = max_len
        self.stride = stride
        self.pool = pool
        self.pad_id = pad_id

        self.sensor_encoder = TimeSeriesTransformer(
            input_dim=6, d_model=self.hidden_size, nhead=4,
            num_encoder_layers=6, dim_feedforward=2048, dropout=0.1
        )

        self.txt_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )
        self.sen_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

        self.logit_scale = nn.Parameter(torch.tensor(0.0)) 

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
        device = input_ids.device
        if L <= self.max_len:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return self._pool_hidden(out.last_hidden_state, attention_mask)  # (B,H)

        chunk_embs = []
        step = self.max_len - self.stride if self.max_len > self.stride else self.max_len
        for b in range(B):
            ids_b = input_ids[b]; mask_b = attention_mask[b]
            parts, weights = [], []
            start = 0
            while start < L:
                end = min(start + self.max_len, L)
                ids_chunk = ids_b[start:end]
                mask_chunk = mask_b[start:end]
                valid_len = mask_chunk.sum().item()

                pad_len = self.max_len - (end - start)
                if pad_len > 0:
                    ids_chunk  = torch.cat([ids_chunk,  torch.full((pad_len,), self.pad_id, dtype=ids_chunk.dtype, device=device)], dim=0)
                    mask_chunk = torch.cat([mask_chunk, torch.zeros(pad_len, dtype=mask_chunk.dtype, device=device)], dim=0)

                ids_chunk = ids_chunk.unsqueeze(0)
                mask_chunk = mask_chunk.unsqueeze(0)

                out = self.bert(input_ids=ids_chunk, attention_mask=mask_chunk)
                emb = self._pool_hidden(out.last_hidden_state, mask_chunk).squeeze(0)  # (H,)
                parts.append(emb); weights.append(valid_len)

                if end == L:
                    break
                start += step

            parts   = torch.stack(parts, dim=0)                                # (C,H)
            weights = torch.tensor(weights, device=device, dtype=parts.dtype)  # (C,)
            weights = (weights / weights.sum().clamp(min=1e-6)).unsqueeze(-1)  # (C,1)
            doc_emb = (parts * weights).sum(dim=0)                              # (H,)
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

        sensor_embeddings  = self.sensor_encoder(time_series)   # (B,T,H)

        text_vec = F.normalize(self.txt_proj(e1), dim=-1)                    
        sensor_vec = F.normalize(self.sen_proj(sensor_embeddings.sum(dim=1)), dim=-1)  # (B,H)

        return (e1, e2,
                sensor_embeddings,
                anchor_embeddings1_1, positive_embeddings2_1, negative_embeddings3_1,
                anchor_embeddings1_2, positive_embeddings2_2, negative_embeddings3_2,
                labels,
                text_vec, sensor_vec, self.logit_scale)



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

def clip_loss(image_features, text_features, logit_scale=None, temperature=None):

    if logit_scale is None:
        assert temperature is not None, "pass logit_scale or temperature"
        scale = 1.0 / max(temperature, 1e-6)
    else:
        scale = logit_scale.exp().clamp(max=100.0)

    logits_per_image = (image_features @ text_features.T) * scale
    logits_per_text  = (text_features  @ image_features.T) * scale
    labels = torch.arange(image_features.size(0), device=image_features.device)
    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_txt = F.cross_entropy(logits_per_text,  labels)
    return 0.5 * (loss_img + loss_txt)




def clip_loss_multipos(
    z_a, z_b, labels, logit_scale=None, temperature=None, eps=1e-12
):

    dev = z_a.device
    z_b = z_b.to(dev)
    labels = labels.to(dev)


    if logit_scale is None:
        assert temperature is not None
        scale = 1.0 / max(float(temperature), 1e-6)
        scale = torch.tensor(scale, device=dev)
    else:

        scale = logit_scale.to(dev).exp().clamp(max=100.0)


    logits_ab = (z_a @ z_b.t()) * scale      
    logits_ba = (z_b @ z_a.t()) * scale   

    same = labels.unsqueeze(1).eq(labels.unsqueeze(0))  
    target_ab = same.float()                      
    target_ba = same.t().float()          


    target_ab = target_ab / (target_ab.sum(dim=1, keepdim=True) + eps)
    target_ba = target_ba / (target_ba.sum(dim=1, keepdim=True) + eps)


    log_q_ab = logits_ab - torch.logsumexp(logits_ab, dim=1, keepdim=True)
    log_q_ba = logits_ba - torch.logsumexp(logits_ba, dim=1, keepdim=True)

    loss_ab = -(target_ab * log_q_ab).sum(dim=1).mean()
    loss_ba = -(target_ba * log_q_ba).sum(dim=1).mean()
    return 0.5 * (loss_ab + loss_ba)
