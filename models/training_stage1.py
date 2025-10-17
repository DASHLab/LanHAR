import argparse
from torch.optim import AdamW
import os

from load_data import *
from model import *
from label_generation import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_save_path", type=str, default="", help="Path to save the trained model")
    parser.add_argument("--beta", type=float, default=0.5, help="")
    parser.add_argument("--source", type=str, default="uci", help="Source dataset name")
    parser.add_argument("--batch_size", type=int, default=10, help="")
    parser.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased", help="")
    parser.add_argument("--num_epochs", type=int, default=10, help="")
    parser.add_argument("--max_len", type=int, default=512, help="")
    parser.add_argument("--stride", type=int, default=128, help="")
    parser.add_argument("--lr", type=float, default=1e-5, help="")

    return parser.parse_args()



def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    model = lanhar(
        bert_model=args.model_name, max_len=args.max_len, stride=args.stride, pool="mean",
        pad_id=pad_id
        ).float().to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    triplet_loss_fn = TripletLoss(margin=1.0)

    bert_params = model.module.bert.parameters() if isinstance(model, nn.DataParallel) else model.bert.parameters()
    optimizer = AdamW(bert_params, lr=args.lr)
    
    dataloader1 = load_data_stage1(args.source, tokenizer, args.batch_size)

    num_epochs = args.num_epochs
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
             input_ids6, attention_mask6) in tqdm(dataloader1):

            optimizer.zero_grad()

            time_series     = time_series.to(device).float()
            input_ids1      = input_ids1.to(device); attention_mask1 = attention_mask1.to(device)
            input_ids2      = input_ids2.to(device); attention_mask2 = attention_mask2.to(device)
            input_ids3      = input_ids3.to(device); attention_mask3 = attention_mask3.to(device)
            input_ids4      = input_ids4.to(device); attention_mask4 = attention_mask4.to(device)
            input_ids5      = input_ids5.to(device); attention_mask5 = attention_mask5.to(device)
            input_ids6      = input_ids6.to(device); attention_mask6 = attention_mask6.to(device)
            labels_dev      = labels.to(device).long()

            outputs = model(
                input_ids1, attention_mask1,
                input_ids2, attention_mask2,
                time_series,
                input_ids3, attention_mask3,
                input_ids4, attention_mask4,
                input_ids5, attention_mask5,
                input_ids6, attention_mask6,
                labels_dev
            )
            (
             embeddings1, embeddings2,
             sensor_embeddings,
             anchor_embeddings1_1, positive_embeddings2_1, negative_embeddings3_1,
             anchor_embeddings1_2, positive_embeddings2_2, negative_embeddings3_2,
             labels_out,
             text_vec, sensor_vec, logit_scale) = outputs

            
            emb_n = F.normalize(embeddings1, dim=-1)  
            #proto_n = F.normalize(embeddings2, dim=-1)

            label_proto_n = label_embedding_generation(device, model, tokenizer, topk=12, temperature=0.07)
            label_proto_n = label_proto_n.to(device)
            proto_n = label_proto_n[labels_out]       
            proto_n = F.normalize(proto_n, dim=-1)

            loss1 = clip_loss_multipos(
                z_a=emb_n, z_b=proto_n, labels=labels_out,
                temperature=0.1
            )

            cls_scale = 30.0  
            logits_cls = (emb_n @ label_proto_n.T) * cls_scale   
            loss_ce = F.cross_entropy(logits_cls, labels_out)
            lam = 0.3
            loss1 = loss1 + lam * loss_ce

            loss2 = 0.5 * triplet_loss_fn(anchor_embeddings1_1, positive_embeddings2_1, negative_embeddings3_1)
            loss3 = 0.5 * triplet_loss_fn(anchor_embeddings1_2, positive_embeddings2_2, negative_embeddings3_2)

            loss = loss1 + loss2 + loss3
            total_loss += float(loss.detach())


            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = torch.argmax(logits_cls, dim=1)  # (B,)
                correct_predictions += (preds == labels_out).sum().item()
                total_predictions   += labels_out.numel()

        avg_loss = total_loss / max(1, len(dataloader1))
        accuracy = correct_predictions / max(1, total_predictions)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            path = os.path.join(args.model_save_path, f"{args.source}_best_model.pth")
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, path)
            print(f"New best model saved to: {path} (acc={accuracy:.4f})")


if __name__ == "__main__":
    main()
