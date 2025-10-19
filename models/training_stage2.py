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
from testing import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_save_path", type=str, default="", help="Path to save the trained model")
    parser.add_argument("--beta", type=float, default=0.5, help="")
    parser.add_argument("--source", type=str, default="uci", help="Source dataset name")
    parser.add_argument("--target", type=str, default="hhar", help="Target dataset name")
    parser.add_argument("--batch_size", type=int, default=256, help="")
    parser.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased", help="")
    parser.add_argument("--lr", type=float, default=4e-5, help="")
    parser.add_argument("--num_epochs", type=int, default=50, help="")
    parser.add_argument("--max_len", type=int, default=512, help="")
    parser.add_argument("--stride", type=int, default=128, help="")

    return parser.parse_args()

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

    dataloader2, _ = load_data_stage2(args.source, args.target, tokenizer, args.batch_size)
    valid_loader, test_loader = load_data_test(args.source, args.target, tokenizer, args.batch_size)

    def load_model_ckpt(m, path):
        if not os.path.exists(path): 
            print(f"[Warn] ckpt not found: {path}, skip loading.")
            return
        state = torch.load(path, map_location=device)
        if len(state) > 0 and next(iter(state.keys())).startswith("module."):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        m.load_state_dict(state, strict=False)
        print(f"[Info] Loaded ckpt from: {path}")

    load_model_ckpt(model, f"{args.source}_best_model.pth")

    for p in model.bert.parameters():
        p.requires_grad = False

    optim_params = (
        list(model.sensor_encoder.parameters())
        + list(model.txt_proj.parameters())
        + list(model.sen_proj.parameters())
        + [model.logit_scale]
    )
    optimizer = AdamW(optim_params, lr = args.lr)
    scaler = GradScaler()

    log_file = os.path.join("", f"training_log_{args.source}_{args.target}.txt")
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logger = logging.getLogger()

    num_epochs = args.num_epochs
    best_accuracy = 0.0
    best_test_accuracy = 0.0
    best_valid_accuracy = 0.0

    for epoch in range(num_epochs):
        with torch.no_grad():
            label_proto_n = label_embedding_generation(device, model, tokenizer, topk=12, temperature=0.07)
            label_proto_n = label_proto_n.to(device)
            label_emb_norm  = F.normalize(model.txt_proj(label_proto_n), dim=-1) 
        
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for (time_series,
             input_ids1, attention_mask1,
             input_ids2, attention_mask2,
             labels) in tqdm(dataloader2):

            optimizer.zero_grad(set_to_none=True)

            time_series     = time_series.to(device, non_blocking=True).float()  # (B,T,6)
            input_ids1      = input_ids1.to(device, non_blocking=True)
            attention_mask1 = attention_mask1.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True) 

            with amp.autocast('cuda'):
                (emb1, emb2,
                 sensor_embeddings,
                 a11, p21, n31,
                 a12, p22, n32,
                 labels_out,
                 text_vec, sensor_vec, logit_scale) = model(
                    input_ids1, attention_mask1,
                    input_ids2.to(device), attention_mask2.to(device),
                    time_series,

                    input_ids1, attention_mask1,
                    input_ids1, attention_mask1,
                    input_ids1, attention_mask1,
                    input_ids1, attention_mask1,
                    labels.to(device)
                )


                proto_n = label_emb_norm[labels_out]       # (B,H)
                proto_n = F.normalize(proto_n, dim=-1)
                loss = clip_loss(sensor_vec, text_vec, logit_scale=logit_scale)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach())


        average_loss = total_loss / max(1, len(dataloader2))
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")
            

        model.eval()
        with torch.no_grad():
            right = 0
            total = 0
            all_labels = []
            all_predictions = []

            for (time_series, input_ids1, attention_mask1, input_ids2, attention_mask2, labels_eval) in valid_loader:
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

            logger.info(f"Cross_dataset_valid: Accuracy = {accuracy:.4f}, F1 = {f1:.4f} | BestAcc = {best_valid_accuracy:.4f}")

            if accuracy > best_valid_accuracy:
                best_valid_accuracy = accuracy
                torch.save(model.state_dict(), os.path.join(args.model_save_path, f"{args.source}_{args.target}_best_model_step2_sensor.pth"))
                logger.info(f"New best model saved")
        





if __name__ == "__main__":
    main()
