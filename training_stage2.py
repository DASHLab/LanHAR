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
from torch import amp
from Our_HAR_read_data import * 
from model import *
from dataset import *


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



    model = Text_SimilarityModel(
        bert_model=model_name,
        max_len=512,
        stride=128,
        pool="mean"
    ).float().to(device)

    def load_model_ckpt(m, path):
        state = torch.load(path, map_location=device)
        if len(state) > 0 and next(iter(state.keys())).startswith("module."):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        m.load_state_dict(state, strict=True)

    load_model_ckpt(model, f"{source}_best_model.pth")

    for p in model.bert.parameters():
        p.requires_grad = False

    optim_params = list(model.sensor_encoder.parameters()) + list(model.sensor_encoder2.parameters())
    optimizer = AdamW(optim_params, lr=4e-5)

    scaler = GradScaler()

    log_file = os.path.join("", f"training_log_{source}_{target}.txt")
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logger = logging.getLogger()

    with torch.no_grad():
        fixed_label_text = generate_label_cross(target_text)  
        vecs = []
        for txt in fixed_label_text:
            enc = tokenizer(txt, add_special_tokens=True, truncation=False, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            emb = model._bert_embed_long(enc["input_ids"], enc["attention_mask"])  # (1,H)
            vecs.append(emb.squeeze(0))
        label_emb = torch.stack(vecs, dim=0).to(device)  # (C,H)
        label_emb_norm = label_emb / (label_emb.norm(dim=-1, keepdim=True) + 1e-12)

    num_epochs = 500
    best_accuracy = 0.0
    best_test_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for (time_series,
             input_ids1, attention_mask1,
             input_ids2, attention_mask2,
             labels) in dataloader2:

            optimizer.zero_grad(set_to_none=True)

            time_series     = time_series.to(device, non_blocking=True).float()  # (B,120,6)
            input_ids1      = input_ids1.to(device, non_blocking=True)
            attention_mask1 = attention_mask1.to(device, non_blocking=True)
            input_ids2      = input_ids2.to(device, non_blocking=True)
            attention_mask2 = attention_mask2.to(device, non_blocking=True)
            labels          = labels.to(device, non_blocking=True)

            with amp.autocast('cuda'):
                (similarity_matrix1, similarity_matrix2,
                 embeddings1, embeddings2,
                 sensor_embeddings, sensor_embeddings2,
                 anchor_embeddings1_1, positive_embeddings2_1, negative_embeddings3_1,
                 anchor_embeddings1_2, positive_embeddings2_2, negative_embeddings3_2,
                 labels_out) = model(
                    input_ids1, attention_mask1,
                    input_ids2, attention_mask2,
                    time_series,
                    input_ids1, attention_mask1,
                    input_ids1, attention_mask1,
                    input_ids1, attention_mask1,
                    input_ids1, attention_mask1,
                    labels
                )


                sensor_vec = sensor_embeddings.sum(dim=1)                                  # (B,H)
                sensor_vec = sensor_vec / (sensor_vec.norm(dim=-1, keepdim=True) + 1e-12)  # (B,H)
                text_vec   = embeddings1 / (embeddings1.norm(dim=-1, keepdim=True) + 1e-12)


                loss = clip_loss(sensor_vec, text_vec)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach())


            with torch.no_grad():
                logits_train = torch.matmul(sensor_vec, text_vec.T)  # (B,B)
                pred_idx = logits_train.argmax(dim=1)
                correct_predictions += (pred_idx == torch.arange(len(labels), device=pred_idx.device)).sum().item()
                total_predictions   += labels.size(0)

        # —— 训练日志
        average_loss = total_loss / max(1, len(dataloader2))
        train_acc = (correct_predictions / max(1, total_predictions)) if total_predictions > 0 else 0.0
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}, Accuracy: {train_acc:.4f}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}, Accuracy: {train_acc:.4f}")

        # —— 保存最佳
        if train_acc > best_accuracy:
            best_accuracy = train_acc
            torch.save(model.state_dict(), os.path.join(model_save_path, f"{source}_{target}_best_model_step2_sensor.pth"))
            print(f"New best model saved with accuracy: {train_acc:.4f}")
            logger.info(f"New best model saved with accuracy: {train_acc:.4f}")

        # —— 评测：传感器 → 固定标签向量（zero-shot 风格），用余弦相似度
        model.eval()
        with torch.no_grad():
            right = 0
            total = 0
            all_labels = []
            all_predictions = []

            for (time_series, input_ids1, attention_mask1, input_ids2, attention_mask2, labels_eval) in dataloader3:
                time_series = time_series.to(device, non_blocking=True).float()

                sensor_embeddings_eval = model.sensor_encoder(time_series)           # (B,T,H)
                sensor_vec_eval = sensor_embeddings_eval.sum(dim=1)                  # (B,H)
                sensor_vec_eval = sensor_vec_eval / (sensor_vec_eval.norm(dim=-1, keepdim=True) + 1e-12)

                # 与固定的 label_emb_norm 做余弦
                logits = torch.matmul(sensor_vec_eval, label_emb_norm.T)             # (B,C)
                preds = torch.argmax(logits, dim=1)

                right += (preds.cpu() == labels_eval).sum().item()
                total += labels_eval.size(0)
                all_labels.extend(labels_eval.numpy().tolist())
                all_predictions.extend(preds.cpu().numpy().tolist())

            accuracy = right / max(1, total)
            try:
                from sklearn.metrics import f1_score
                f1 = f1_score(all_labels, all_predictions, average='weighted')
            except Exception:
                f1 = 0.0

            print(f"Cross_dataset: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")
            logger.info(f"Cross_dataset: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")

            if accuracy > best_test_accuracy:
                best_test_accuracy = accuracy
                df = pd.DataFrame({'labels': all_labels, 'predictions': all_predictions})
                df.to_csv(os.path.join(model_save_path, f'cross_dataset_labels_predictions_{source}_{target}.csv'), index=False)

            print(f"Cross_dataset: best Accuracy = {best_test_accuracy:.4f}")
            logger.info(f"Cross_dataset: best Accuracy = {best_test_accuracy:.4f}")