"""Train one DeBERTa-v3-base on ALL of train.json, then label test.json.

Final deployment step: no CV, no held-out split. Saves the model to deberta_final/
and writes submission.csv (reviews,predictions) in test.json order.
Run from the project root.

Usage:
    python models/deberta_final.py
    python models/deberta_final.py --model-name microsoft/deberta-v3-base
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_cv import ReviewDataset, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--model-name", default="microsoft/deberta-v3-base")
    ap.add_argument("--out-dir", default="deberta_final")
    ap.add_argument("--submission", default="submission.csv")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8")) if os.path.exists(args.config) else {}
    g = lambda k, d: cfg.get(k, d)
    epochs = int(g("epochs", 3))
    batch_size = int(g("batch_size", 32))
    eval_batch_size = int(g("eval_batch_size", 64))
    lr = float(g("lr", 2e-5))
    max_len = int(g("max_len", 128))
    weight_decay = float(g("weight_decay", 0.01))
    warmup_frac = float(g("warmup_frac", 0.1))
    grad_clip = float(g("grad_clip", 1.0))
    threshold = float(g("threshold", 0.5))
    seed = int(g("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = bool(g("fp16", True)) and device.type == "cuda"
    print(f"device={device} model={args.model_name}")
    print(f"epochs={epochs} batch={batch_size} lr={lr} max_len={max_len} threshold={threshold} seed={seed} fp16={use_fp16}")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train = json.load(open("train.json", encoding="utf-8"))
    texts = [r["reviews"] for r in train]
    labels = [int(r["sentiments"]) for r in train]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    enc = tokenizer(texts, truncation=True, max_length=max_len)
    collator = DataCollatorWithPadding(tokenizer)
    ds = ReviewDataset(enc["input_ids"], enc["attention_mask"], labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    print(f"train={len(ds)} batches_per_epoch={len(loader)}")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).float().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * warmup_frac), total_steps)
    scaler = torch.amp.GradScaler(enabled=use_fp16)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(batch["labels"])
        print(f"epoch {epoch}: loss={total / len(ds):.4f} ({time.time() - t0:.1f}s)")

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"saved model + tokenizer to {args.out_dir}/")

    test = json.load(open("test.json", encoding="utf-8"))
    test_texts = [r["reviews"] for r in test]
    enc_te = tokenizer(test_texts, truncation=True, max_length=max_len)
    te_ds = ReviewDataset(enc_te["input_ids"], enc_te["attention_mask"], [0] * len(test_texts))
    te_loader = DataLoader(te_ds, batch_size=eval_batch_size, collate_fn=collator)
    probs = predict(model, te_loader, device)
    pred = (probs >= threshold).astype(int)

    assert len(pred) == len(test_texts) == 1851, f"row count mismatch: {len(pred)} vs {len(test_texts)}"
    assert all(t.strip() for t in test_texts), "empty review found"

    tmp = args.submission + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reviews", "predictions"])
        for text, p in zip(test_texts, pred):
            w.writerow([text, int(p)])
    os.replace(tmp, args.submission)

    with open(args.submission, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1851, f"written rows: {len(rows)}"
    assert list(rows[0].keys()) == ["reviews", "predictions"]
    assert all(r["reviews"] == t for r, t in zip(rows, test_texts)), "text order mismatch"

    print(f"wrote {args.submission}: {len(pred)} rows, positives={int(pred.sum())}, negatives={int((1 - pred).sum())}")
    print(f"probability summary: min={probs.min():.4f} mean={probs.mean():.4f} max={probs.max():.4f}")


if __name__ == "__main__":
    main()
