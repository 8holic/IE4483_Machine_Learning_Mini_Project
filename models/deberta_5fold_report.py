"""DeBERTa-only 5-fold CV report + test confidence report.

Stage cv (default):
    5-fold CV on train.json. Writes results/deberta_5fold_predictions.csv with
    columns: reviews, prediction, accuracy, confidence, prob_positive, true_label, fold.
    `accuracy` is row-level: 1 if the prediction matches the held-out label, else 0.
    `confidence` is max(prob_positive, 1 - prob_positive).
    Prints per-fold accuracy and pooled accuracy.

Stage test:
    Loads one saved DeBERTa model (default deberta_final/), scores test.json, and writes
    results/deberta_test_predictions.csv with columns:
    reviews, prediction, confidence, prob_positive.
    No accuracy column: test labels are hidden.

Nothing is written to submission.csv. Run from the project root.

Usage:
    python models/deberta_5fold_report.py                  # 5-fold CSV first
    python models/deberta_5fold_report.py --stage test     # test CSV
    python models/deberta_5fold_report.py --stage all      # both
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_cv import ReviewDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs = []
    with torch.jit.optimized_execution(False):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def train_fold(model_name, tr_ds, tr_loader, args, device, fold_seed):
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    torch.cuda.manual_seed_all(fold_seed)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).float().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(tr_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * args.warmup_frac), total_steps)
    use_fp16 = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_fp16)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for batch in tr_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.jit.optimized_execution(False):
                with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
                    loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(batch["labels"])
        print(f"  epoch {epoch}: loss={total / len(tr_ds):.4f} ({time.time() - t0:.1f}s)")

    return model


def run_cv(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_seed = args.seed
    train = json.load(open(args.data, encoding="utf-8"))
    texts = [r["reviews"] for r in train]
    labels = np.array([int(r["sentiments"]) for r in train])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    enc = tokenizer(texts, truncation=True, max_length=args.max_len)
    collator = DataCollatorWithPadding(tokenizer)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows = []
    fold_accs = []

    for k, (tr_idx, te_idx) in enumerate(skf.split(texts, labels), 1):
        t0 = time.time()
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

        model = train_fold(args.model_name, tr_ds, tr_loader, args, device)
        probs = predict_probs(model, te_loader, device)
        pred = (probs >= args.threshold).astype(int)
        true = labels[te_idx]
        correct = (pred == true).astype(int)
        conf = np.maximum(probs, 1 - probs)

        acc = accuracy_score(true, pred)
        f1 = f1_score(true, pred)
        fold_accs.append(acc)
        print(
            f"fold {k}: accuracy={acc:.4f} f1={f1:.4f} mean_confidence={conf.mean():.4f} "
            f"({time.time() - t0:.1f}s)"
        )

        for i, j in enumerate(te_idx):
            rows.append({
                "reviews": texts[j],
                "prediction": int(pred[i]),
                "accuracy": int(correct[i]),
                "confidence": float(conf[i]),
                "prob_positive": float(probs[i]),
                "true_label": int(true[i]),
                "fold": k,
            })

    with open(args.out_cv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["reviews", "prediction", "accuracy", "confidence", "prob_positive", "true_label", "fold"],
        )
        w.writeheader()
        w.writerows(rows)

    pooled_acc = float(np.mean([r["accuracy"] for r in rows]))
    print(f"\nwrote {args.out_cv}: {len(rows)} rows")
    print(f"per-fold accuracy: " + " ".join(f"{a:.4f}" for a in fold_accs))
    print(f"pooled out-of-fold accuracy: {pooled_acc:.4f}")


def run_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.test_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.test_model_dir).float().to(device)

    rows = json.load(open(args.test_data, encoding="utf-8"))
    texts = [r["reviews"] for r in rows]
    enc = tokenizer(texts, truncation=True, max_length=args.max_len)
    ds = ReviewDataset(enc["input_ids"], enc["attention_mask"], [0] * len(texts))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.eval_batch_size, collate_fn=DataCollatorWithPadding(tokenizer))
    probs = predict_probs(model, loader, device)

    pred = (probs >= args.threshold).astype(int)
    conf = np.maximum(probs, 1 - probs)

    with open(args.out_test, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reviews", "prediction", "confidence", "prob_positive"])
        for text, p, c, pr in zip(texts, pred, conf, probs):
            w.writerow([text, int(p), f"{c:.6f}", f"{pr:.6f}"])

    print(f"wrote {args.out_test}: {len(pred)} rows (model={args.test_model_dir}, max_len={args.max_len})")
    print(f"predicted positive: {int(pred.sum())}, negative: {int((1 - pred).sum())}")
    print(f"confidence: min={conf.min():.4f} mean={conf.mean():.4f} median={np.median(conf):.4f} max={conf.max():.4f}")


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config.json")
    cfg_args, _ = pre.parse_known_args()
    cfg = json.load(open(cfg_args.config, encoding="utf-8")) if os.path.exists(cfg_args.config) else {}

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cv", "test", "all"], default="cv")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--model-name", default="microsoft/deberta-v3-base")
    ap.add_argument("--data", default="train.json")
    ap.add_argument("--test-data", default="test.json")
    ap.add_argument("--test-model-dir", default="deberta_final")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-cv", default="results/deberta_5fold_predictions.csv")
    ap.add_argument("--out-test", default="results/deberta_test_predictions.csv")
    known = {a.dest for a in ap._actions}
    ap.set_defaults(**{k: v for k, v in cfg.items() if k in known})
    args = ap.parse_args()

    if args.stage in ("cv", "all"):
        run_cv(args, cfg)
    if args.stage in ("test", "all"):
        run_test(args)


if __name__ == "__main__":
    main()
