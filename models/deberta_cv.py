"""DeBERTa-v3-base 5-fold CV with one prediction CSV per fold.

For each fold, trains on 4/5 of train.json, predicts the held-out fold, and writes:
    results/deberta_fold_<k>_predictions.csv
with columns:
    reviews, prediction, accuracy, confidence, true_label

`prediction` is the guessed label (0/1).
`accuracy` is 1 when the guess matches true_label, else 0.
`confidence` is max(P(positive), 1 - P(positive)).
`true_label` is the correct held-out label.

Run from the project root:
    python models/deberta_cv.py
    python models/deberta_cv.py --epochs 4 --batch-size 16
    python models/deberta_cv.py --max-len 256 --only-fold 2 --out-prefix results/deberta_256_fold
    python models/deberta_cv.py --max-len 256 --batch-size 16 --grad-accum 2 --probe
    python models/deberta_cv.py --max-len 256 --batch-size 16 --grad-accum 2 --only-fold 2 --out-prefix results/deberta_256_fold
"""

import argparse
import csv
import json
import math
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


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config.json")
    cfg_args, _ = pre.parse_known_args()
    cfg = json.load(open(cfg_args.config, encoding="utf-8")) if os.path.exists(cfg_args.config) else {}

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--model-name", default="microsoft/deberta-v3-base")
    ap.add_argument("--data", default="train.json")
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
    ap.add_argument("--out-prefix", default="results/deberta_fold")
    ap.add_argument("--only-fold", type=int, default=None, help="run only this 1-based fold")
    ap.add_argument("--grad-accum", type=int, default=1, help="accumulate gradients over this many microbatches")
    ap.add_argument("--probe", action="store_true", help="run one padded step, print peak VRAM, then exit")
    known = {a.dest for a in ap._actions}
    ap.set_defaults(**{k: v for k, v in cfg.items() if k in known})
    args = ap.parse_args()

    if args.only_fold is not None and not (1 <= args.only_fold <= args.folds):
        ap.error(f"--only-fold must be between 1 and {args.folds}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} model={args.model_name}")
    print(
        f"folds={args.folds} epochs={args.epochs} batch={args.batch_size} "
        f"grad_accum={args.grad_accum} max_len={args.max_len} threshold={args.threshold} seed={args.seed}"
    )

    train = json.load(open(args.data, encoding="utf-8"))
    texts = [r["reviews"] for r in train]
    labels = np.array([int(r["sentiments"]) for r in train])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    enc = tokenizer(texts, truncation=True, max_length=args.max_len)
    collator = DataCollatorWithPadding(tokenizer)
    print(f"reviews={len(texts)}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    for k, (tr_idx, te_idx) in enumerate(skf.split(texts, labels), 1):
        if args.only_fold is not None and k != args.only_fold:
            continue
        t0 = time.time()
        fold_seed = args.seed + k
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        torch.cuda.manual_seed_all(fold_seed)

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
        total_steps = math.ceil(len(tr_loader) / args.grad_accum) * args.epochs
        sched = get_linear_schedule_with_warmup(opt, int(total_steps * args.warmup_frac), total_steps)
        use_fp16 = args.fp16 and device.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_fp16)

        if args.probe:
            first = next(iter(tr_loader))
            ids, mask = first["input_ids"], first["attention_mask"]
            pad_len = args.max_len - ids.size(1)
            if pad_len > 0:
                ids = torch.cat([ids, torch.zeros((ids.size(0), pad_len), dtype=ids.dtype)], dim=1)
                mask = torch.cat([mask, torch.zeros((mask.size(0), pad_len), dtype=mask.dtype)], dim=1)
            batch = {"input_ids": ids.to(device), "attention_mask": mask.to(device), "labels": first["labels"].to(device)}
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            opt.zero_grad()
            with torch.jit.optimized_execution(False):
                with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
                    loss = model(**batch).loss
            scaler.scale(loss / args.grad_accum).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()
                free, total = torch.cuda.mem_get_info()
                print(
                    f"probe: seq_len={batch['input_ids'].size(1)} micro_batch={batch['input_ids'].size(0)} "
                    f"peak_allocated={torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB "
                    f"peak_reserved={torch.cuda.max_memory_reserved() / 1024**2:.0f} MiB "
                    f"free_after={free / 1024**2:.0f} MiB total={total / 1024**2:.0f} MiB"
                )
            else:
                print("probe: CPU run, no VRAM stats")
            return

        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            opt.zero_grad()
            for step, batch in enumerate(tr_loader, 1):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.jit.optimized_execution(False):
                    with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
                        loss = model(**batch).loss
                total += loss.item() * len(batch["labels"])
                scaler.scale(loss / args.grad_accum).backward()
                if step % args.grad_accum == 0 or step == len(tr_loader):
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                    sched.step()
            print(f"  fold {k} epoch {epoch}: loss={total / len(tr_ds):.4f}")

        probs = predict_probs(model, te_loader, device)
        pred = (probs >= args.threshold).astype(int)
        true = labels[te_idx]
        conf = np.maximum(probs, 1 - probs)
        correct = (pred == true).astype(int)

        out_path = f"{args.out_prefix}_{k}_predictions.csv"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["reviews", "prediction", "accuracy", "confidence", "true_label"])
            for i, j in enumerate(te_idx):
                w.writerow([texts[j], int(pred[i]), int(correct[i]), f"{conf[i]:.6f}", int(true[i])])

        print(
            f"fold {k}: accuracy={accuracy_score(true, pred):.4f} "
            f"f1={f1_score(true, pred):.4f} mean_confidence={conf.mean():.4f} "
            f"-> {out_path} ({time.time() - t0:.1f}s)"
        )


if __name__ == "__main__":
    main()
