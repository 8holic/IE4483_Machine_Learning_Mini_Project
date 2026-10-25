"""5-fold cross-validation of a BiLSTM sentiment classifier on train.json.

Reads only train.json, writes results/cv_results.png and results/threshold_sweep.png.
Fixed epoch count, fixed seed, vocabulary rebuilt per fold. Run from the project root.

Usage:
    python models/bilstm_cv.py
    python models/bilstm_cv.py --folds 5 --epochs 7
"""

import argparse
import json
import random
import re
import time
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

PAD, UNK = 0, 1


def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def build_vocab(token_lists, min_freq=2, max_size=30000):
    counts = Counter(tok for toks in token_lists for tok in toks)
    words = sorted((w for w, c in counts.items() if c >= min_freq), key=counts.get, reverse=True)
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for w in words[: max_size - 2]:
        vocab[w] = len(vocab)
    return vocab


def encode(tokens, vocab, max_len):
    ids = [vocab.get(t, UNK) for t in tokens][:max_len]
    return ids or [UNK]


def load_glove(path, dim):
    vectors = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) == dim + 1:
                vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return vectors


def embedding_matrix(vocab, glove, dim, seed):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    m[PAD] = 0
    hits = 0
    for w, i in vocab.items():
        v = glove.get(w)
        if v is not None:
            m[i] = v
            hits += 1
    return torch.from_numpy(m), hits


def make_batches(ids_list, labels, batch_size, shuffle, rng=None):
    order = list(range(len(ids_list)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = order[i : i + batch_size]
        length = max(len(ids_list[j]) for j in chunk)
        x = torch.zeros(len(chunk), length, dtype=torch.long)
        for r, j in enumerate(chunk):
            x[r, : len(ids_list[j])] = torch.tensor(ids_list[j])
        y = torch.tensor([labels[j] for j in chunk], dtype=torch.float)
        yield x, y


class Classifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 4, 1)

    def forward(self, x):
        mask = (x != PAD).unsqueeze(-1)
        h, _ = self.lstm(self.emb(x))
        h = h.masked_fill(~mask, 0)
        mean = h.sum(1) / mask.sum(1).clamp(min=1)
        mx = h.masked_fill(~mask, -1e4).max(1).values
        return self.fc(self.drop(torch.cat([mean, mx], -1))).squeeze(-1)


@torch.no_grad()
def predict(model, ids_list, batch_size, device):
    model.eval()
    probs = []
    for x, _ in make_batches(ids_list, [0] * len(ids_list), batch_size, False):
        probs.append(torch.sigmoid(model(x.to(device))).cpu())
    return torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--glove", default="glove.6B.100d.txt")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train = json.load(open("train.json", encoding="utf-8"))
    tokens = [tokenize(r["reviews"]) for r in train]
    labels = np.array([int(r["sentiments"]) for r in train])
    print(f"reviews={len(tokens)} folds={args.folds} epochs={args.epochs}")

    glove = load_glove(args.glove, args.emb_dim) if args.glove else None
    if glove is not None:
        print(f"glove: {len(glove)} vectors of dim {args.emb_dim}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    results, all_true, all_pred, all_probs, curves = [], [], [], [], []

    for k, (tr_idx, te_idx) in enumerate(skf.split(tokens, labels), 1):
        t0 = time.time()
        random.seed(args.seed + k)
        np.random.seed(args.seed + k)
        torch.manual_seed(args.seed + k)
        torch.cuda.manual_seed_all(args.seed + k)

        tr_tok = [tokens[i] for i in tr_idx]
        te_tok = [tokens[i] for i in te_idx]
        y_tr = labels[tr_idx].tolist()
        y_te = labels[te_idx].tolist()

        vocab = build_vocab(tr_tok, args.min_freq)
        ids_tr = [encode(t, vocab, args.max_len) for t in tr_tok]
        ids_te = [encode(t, vocab, args.max_len) for t in te_tok]

        model = Classifier(len(vocab), args.emb_dim, args.hidden, args.dropout)
        if glove is not None:
            matrix, hits = embedding_matrix(vocab, glove, args.emb_dim, args.seed + k)
            model.emb.weight.data.copy_(matrix)
            print(f"  glove coverage {hits}/{len(vocab)} ({hits / len(vocab):.1%})")
        model = model.to(device)
        pos_weight = torch.tensor([(len(y_tr) - sum(y_tr)) / sum(y_tr)], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        rng = random.Random(args.seed + k)

        losses, f1_curve = [], []
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for x, y in make_batches(ids_tr, y_tr, args.batch_size, True, rng):
                opt.zero_grad()
                loss = loss_fn(model(x.to(device)), y.to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(y)
            losses.append(total / len(y_tr))
            f1_curve.append(f1_score(y_te, (predict(model, ids_te, args.batch_size, device) >= 0.5).astype(int)))
        curves.append((losses, f1_curve))

        probs = predict(model, ids_te, args.batch_size, device)
        pred = (probs >= 0.5).astype(int)
        m = {
            "acc": accuracy_score(y_te, pred),
            "prec": precision_score(y_te, pred),
            "rec": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
        }
        results.append(m)
        all_true.extend(y_te)
        all_pred.extend(pred)
        all_probs.extend(probs)
        print(
            f"fold {k}: train={len(y_tr)} test={len(y_te)} vocab={len(vocab)} loss={total / len(y_tr):.4f} "
            f"acc={m['acc']:.4f} prec={m['prec']:.4f} rec={m['rec']:.4f} f1={m['f1']:.4f} "
            f"({time.time() - t0:.1f}s)"
        )

    print("\nper-fold F1:", " ".join(f"{m['f1']:.4f}" for m in results))
    for key, name in [("acc", "accuracy"), ("prec", "precision"), ("rec", "recall"), ("f1", "F1")]:
        vals = np.array([m[key] for m in results])
        print(f"{name:9s} mean={vals.mean():.4f} std={vals.std():.4f} min={vals.min():.4f} max={vals.max():.4f}")
    print("pooled confusion matrix (rows=true 0/1, cols=pred 0/1):")
    cm = confusion_matrix(all_true, all_pred)
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
    print("(threshold picked on the same out-of-fold predictions; mildly optimistic)")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(ts, f1_ts, label="F1")
    ax2.plot(ts, [precision_score(true, (probs >= t).astype(int), zero_division=0) for t in ts], label="precision")
    ax2.plot(ts, [recall_score(true, (probs >= t).astype(int)) for t in ts], label="recall")
    ax2.axvline(bt, color="k", ls="--", label=f"best={bt:.2f}")
    ax2.axvline(0.5, color="grey", ls=":", label="0.5")
    ax2.set(title="Threshold sweep (pooled out-of-fold)", xlabel="threshold", ylabel="score")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("results/threshold_sweep.png", dpi=130)
    print("saved results/threshold_sweep.png")

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for k, (losses, f1_curve) in enumerate(curves, 1):
        ax[0, 0].plot(range(1, len(losses) + 1), losses, label=f"fold {k}")
        ax[0, 1].plot(range(1, len(f1_curve) + 1), f1_curve, label=f"fold {k}")
    ax[0, 0].set(title="Training loss (weighted BCE)", xlabel="epoch", ylabel="loss")
    ax[0, 1].set(title="Held-out F1", xlabel="epoch", ylabel="F1")
    ax[0, 0].legend()
    ax[0, 1].legend()
    f1s = [m["f1"] for m in results]
    ax[1, 0].bar(range(1, len(f1s) + 1), f1s)
    ax[1, 0].axhline(np.mean(f1s), color="r", ls="--", label=f"mean={np.mean(f1s):.4f}")
    ax[1, 0].set(title="F1 per fold", xlabel="fold", ylabel="F1", ylim=(0.85, 1.0))
    ax[1, 0].legend()
    ax[1, 1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[1, 1].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13)
    ax[1, 1].set(title="Pooled confusion matrix", xticks=[0, 1], yticks=[0, 1], xlabel="predicted", ylabel="true")
    fig.tight_layout()
    fig.savefig("results/cv_results.png", dpi=130)
    print("saved results/cv_results.png")


if __name__ == "__main__":
    main()
