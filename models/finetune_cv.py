"""Fine-tune a HuggingFace sequence classifier with 5-fold CV on train.json.

Works for any AutoModelForSequenceClassification checkpoint (BERT, DeBERTa, ...).
Reads only train.json. Output plot name is configurable via --out. Run from the project root.
Standard recipe: lr 2e-5, 3 epochs, batch 32, max_len 128, 10% warmup, weight decay 0.01.

Usage:
    python models/finetune_cv.py                                  # bert-base-uncased
    python models/finetune_cv.py --model-name microsoft/deberta-v3-base --out results/deberta_results.png
    python models/finetune_cv.py --epochs 2 --folds 3
"""

import argparse
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="bert-base-uncased")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/bert_results.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} model={args.model_name}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    train = json.load(open("train.json", encoding="utf-8"))
    texts = [r["reviews"] for r in train]
    labels = np.array([int(r["sentiments"]) for r in train])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    enc = tokenizer(texts, truncation=True, max_length=args.max_len)
    collator = DataCollatorWithPadding(tokenizer)
    print(f"reviews={len(texts)} folds={args.folds} epochs={args.epochs} max_len={args.max_len}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    results, all_true, all_pred, all_probs, curves = [], [], [], [], []

    for k, (tr_idx, te_idx) in enumerate(skf.split(texts, labels), 1):
        t0 = time.time()
        random.seed(args.seed + k)
        np.random.seed(args.seed + k)
        torch.manual_seed(args.seed + k)
        torch.cuda.manual_seed_all(args.seed + k)

        tr_ds = ReviewDataset(
            [enc["input_ids"][i] for i in tr_idx],
            [enc["attention_mask"][i] for i in tr_idx],
            labels[tr_idx].tolist(),
        )
        te_ds = ReviewDataset(
            [enc["input_ids"][i] for i in te_idx],
            [enc["attention_mask"][i] for i in te_idx],
            labels[te_idx].tolist(),
        )
        tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
        te_loader = torch.utils.data.DataLoader(te_ds, batch_size=args.eval_batch_size, collate_fn=collator)

        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).float().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        total_steps = len(tr_loader) * args.epochs
        sched = get_linear_schedule_with_warmup(opt, int(total_steps * args.warmup_frac), total_steps)
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

        losses = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for batch in tr_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad()
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss = model(**batch).loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                total += loss.item() * len(batch["labels"])
            losses.append(total / len(tr_ds))
            probs = predict(model, te_loader, device)
            print(f"  fold {k} epoch {epoch}: loss={losses[-1]:.4f} f1={f1_score(labels[te_idx], (probs >= 0.5).astype(int)):.4f}")
        curves.append(losses)

        probs = predict(model, te_loader, device)
        pred = (probs >= 0.5).astype(int)
        m = {
            "acc": accuracy_score(labels[te_idx], pred),
            "prec": precision_score(labels[te_idx], pred),
            "rec": recall_score(labels[te_idx], pred),
            "f1": f1_score(labels[te_idx], pred),
        }
        results.append(m)
        all_true.extend(labels[te_idx].tolist())
        all_pred.extend(pred.tolist())
        all_probs.extend(probs)
        print(
            f"fold {k}: acc={m['acc']:.4f} prec={m['prec']:.4f} rec={m['rec']:.4f} f1={m['f1']:.4f} "
            f"({time.time() - t0:.1f}s)"
        )

    print("\nper-fold F1:", " ".join(f"{m['f1']:.4f}" for m in results))
    for key, name in [("acc", "accuracy"), ("prec", "precision"), ("rec", "recall"), ("f1", "F1")]:
        vals = np.array([m[key] for m in results])
        print(f"{name:9s} mean={vals.mean():.4f} std={vals.std():.4f} min={vals.min():.4f} max={vals.max():.4f}")
    cm = confusion_matrix(all_true, all_pred)
    print("pooled confusion matrix (rows=true 0/1, cols=pred 0/1):")
    print(cm)

    probs = np.array(all_probs)
    true = np.array(all_true)
    ts = np.arange(0.10, 0.91, 0.01)
    f1_ts = [f1_score(true, (probs >= t).astype(int)) for t in ts]
    bi = int(np.argmax(f1_ts))
    bt = float(ts[bi])
    pt = (probs >= bt).astype(int)
    print(
        f"\npooled OOF threshold sweep: best thr={bt:.2f} f1={f1_ts[bi]:.4f} "
        f"acc={accuracy_score(true, pt):.4f} prec={precision_score(true, pt):.4f} rec={recall_score(true, pt):.4f}"
    )

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, losses in enumerate(curves, 1):
        ax[0].plot(range(1, len(losses) + 1), losses, label=f"fold {k}")
    ax[0].set(title="Training loss", xlabel="epoch", ylabel="loss")
    ax[0].legend()
    f1s = [m["f1"] for m in results]
    ax[1].bar(range(1, len(f1s) + 1), f1s)
    ax[1].axhline(np.mean(f1s), color="r", ls="--", label=f"mean={np.mean(f1s):.4f}")
    ax[1].set(title="F1 per fold", xlabel="fold", ylabel="F1", ylim=(0.85, 1.0))
    ax[1].legend()
    ax[2].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[2].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13)
    ax[2].set(title="Pooled confusion matrix", xticks=[0, 1], yticks=[0, 1], xlabel="predicted", ylabel="true")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
