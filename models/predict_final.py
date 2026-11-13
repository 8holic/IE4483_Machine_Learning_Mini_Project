"""Label test.json with the trained checkpoint in deberta_final/ and write submission_final.csv.

Inference only (no training). Run from the project root:
    python models/predict_final.py
"""

import argparse
import csv
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_cv import ReviewDataset, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="deberta_final")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--test", default="test.json")
    ap.add_argument("--out", default="submission_final.csv")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8")) if os.path.exists(args.config) else {}
    max_len = args.max_len or int(cfg.get("max_len", 128))
    threshold = args.threshold if args.threshold is not None else float(cfg.get("threshold", 0.5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, num_labels=2).float().to(device)
    collator = DataCollatorWithPadding(tokenizer)

    test = json.load(open(args.test, encoding="utf-8"))
    texts = [r["reviews"] for r in test]
    enc = tokenizer(texts, truncation=True, max_length=max_len)
    ds = ReviewDataset(enc["input_ids"], enc["attention_mask"], [0] * len(texts))
    loader = DataLoader(ds, batch_size=args.eval_batch_size, collate_fn=collator)

    probs = predict(model, loader, device)
    pred = (probs >= threshold).astype(int)

    assert len(pred) == len(texts) == 1851, f"row count mismatch: {len(pred)} vs {len(texts)}"
    assert all(t.strip() for t in texts), "empty review found"

    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reviews", "predictions"])
        for text, p in zip(texts, pred):
            w.writerow([text, int(p)])
    os.replace(tmp, args.out)

    with open(args.out, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1851 and list(rows[0].keys()) == ["reviews", "predictions"]
    assert all(r["reviews"] == t for r, t in zip(rows, texts)), "text order mismatch"

    print(f"device={device} max_len={max_len} threshold={threshold}")
    print(f"wrote {args.out}: {len(pred)} rows, positives={int(pred.sum())}, negatives={int((1 - pred).sum())}")
    print(f"probability summary: min={probs.min():.4f} mean={probs.mean():.4f} max={probs.max():.4f}")


if __name__ == "__main__":
    main()
