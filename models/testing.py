import argparse
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch import amp
import logging, os
import pandas as pd

from load_data import *
from model import *
from label_generation import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_save_path", type=str, default="", help="Path to save the trained model")
    parser.add_argument("--source", type=str, default="uci", help="Source dataset name")
    parser.add_argument("--target", type=str, default="hhar", help="Target dataset name")
    parser.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased", help="")
    parser.add_argument("--batch_size", type=int, default=256, help="")
    parser.add_argument("--max_len", type=int, default=512, help="")
    parser.add_argument("--stride", type=int, default=128, help="")

    return parser.parse_args()


def test(model, device, tokenizer, test_loader):
    model.eval()
    with torch.no_grad():
        label_proto_n = label_embedding_generation(device, model, tokenizer, topk=12, temperature=0.07)
        label_proto_n = label_proto_n.to(device)
        label_emb_norm  = F.normalize(model.txt_proj(label_proto_n), dim=-1) 
        right = 0
        total = 0
        all_labels = []
        all_predictions = []

        for (time_series, input_ids1, attention_mask1, input_ids2, attention_mask2, labels_eval) in test_loader:
            time_series = time_series.to(device, non_blocking=True).float()
            
            sensor_embeddings_eval = model.sensor_encoder(time_series)   
            sensor_vec_eval = F.normalize(model.sen_proj(sensor_embeddings_eval.sum(dim=1)), dim=-1)  
  
            logits = torch.matmul(sensor_vec_eval, label_emb_norm.T)    
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

        print(f"Cross_dataset_test: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
    


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    model = lanhar(
        bert_model=args.model_name,
        max_len=args.max_len,
        stride=args.stride,
        pool="mean",
        pad_id=pad_id
    ).float().to(device)

    _, test_loader = load_data_test(args.source, args.target, tokenizer, args.batch_size)

    def load_model_ckpt(m, path):
        if not os.path.exists(path): 
            print(f"[Warn] ckpt not found: {path}, skip loading.")
            return
        state = torch.load(path, map_location=device)
        if len(state) > 0 and next(iter(state.keys())).startswith("module."):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        m.load_state_dict(state, strict=False)
        print(f"[Info] Loaded ckpt from: {path}")

    load_model_ckpt(model, f"{args.source}_{args.target}_best_model_step2_sensor.pth")        
        
    test(model, device, tokenizer, test_loader)


if __name__ == "__main__":
    main()
