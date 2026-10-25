# Decision log

Project: IE4483 Mini Project 1  Eproduct-review sentiment classification (Amazon reviews).
Files under discussion: `train.json` (7,401 labeled), `test.json` (1,851 unlabeled), `submission.csv` (deliverable).

## Hard constraints (from user)
- One neural model only. No model mixing, no ensembles.
- `test.json` must not be read/used for training or evaluation.
- `submission.csv` must not be touched.
- No logistic regression / non-neural classifiers.
- Cross-validation over `train.json` is the agreed evaluation method.

## Environment
- `.venv` inside the project folder (user's choice).
- Python 3.14.6, torch 2.14.1 **CPU-only**, scikit-learn 1.9.1, numpy 2.4.4, matplotlib 3.11.2.
- GPU (RTX 4060) currently unused: CUDA torch install was aborted; CPU gives ~9 s/epoch, acceptable.

## Data facts (measured)
- Labels: 6,319 positive / 1,082 negative (85.4% / 14.6%).
- Tokens: median 42, mean 57, p95 144 words. Only 1.18% exceed 256 tokens.
- Vocabulary at `min_freq=2`: ~7,000 types; 46.7% of word types occur once; held-out OOV rate 2.70%.
- Text is clean: 0 HTML tags/entities, 2 URLs, 18 non-ASCII reviews, 1,224 contain newlines (harmless).
- Some reviews are truncated mid-word at the source (dataset artifact), e.g. "...good pric". Unrecoverable.

## Decisions
| Decision | Why |
|---|---|
| Tokenize: lowercase, regex `[a-z0-9']+`, keep apostrophes | Keeps negations (`don't`) intact; no stemming/stop-words because function words carry sentiment |
| Vocab built per fold from training fold only, `min_freq=2`, cap 30k | Prevents leakage from held-out reviews; singletons are mostly typos/noise |
| Zero padding + `<unk>` ID 1 | Fixed-length batches; unseen words share one learned vector |
| `max_len=256` | p95 is 144, so 256 covers ~all reviews; 256 chosen as safe headroom. Could drop to ~160 for speed with negligible loss |
| `StratifiedKFold(5, shuffle=True, seed=42)` | Preserves 85/15 class ratio per fold; every review tested once, trained on 4 times; fixed seed for reproducibility |
| Fixed 7 epochs for all folds, no early stopping | Avoids tuning on the held-out fold, which would make the CV score optimistic. 7 came from the earlier single-split run (acknowledged mild tuning) |
| Fresh model per fold, `torch.manual_seed(seed+fold)` | Independent folds, reproducible init/shuffle |
| Model: Embedding(128) ↁEBiLSTM(128/direction) ↁEmasked mean+max pool ↁEDropout(0.4) ↁELinear(512ↁE) | Matches the assignment's "RNN + linear layer"; BiLSTM handles word order and contrast ("great, but broke"); mean+max pooling summarizes variable-length reviews; ~1.16M params appropriate for 5.9k samples |
| Random-initialised embeddings (not pretrained) | superseded | Replaced by GloVe 100d: random vectors learn little for the 47% of word types seen once |
| GloVe 6B 100d initialisation, fine-tuned during training | adopted | 93.5% of vocab covered; +0.6 F1 over random init for one download and no architecture change. Missing words stay random N(0, 0.1); `<pad>` row zeroed |
| Decision threshold reported at 0.5 per fold, plus a pooled out-of-fold sweep | adopted | 0.5 keeps folds comparable; the sweep shows where F1 peaks, flagged as mildly optimistic because it is picked on the same OOF predictions |
| `BCEWithLogitsLoss` with `pos_weight ≁E5.8` (neg/pos) | 85/15 imbalance; without it, predicting all-positive is near-optimal |
| Adam, `lr=1e-3`, batch 64, grad-clip 1.0 | Standard, stable for embedding+LSTM; clipping prevents LSTM exploding gradients |
| Decision threshold fixed at 0.5 | Comparable across folds; tuning per fold on the held-out data would be optimistic. Threshold tuning is an open item |
| Report accuracy, precision, recall, F1, confusion matrix, per-fold mean ± std | F1 is the honest headline under imbalance; std shows split stability |

## Experiments
| Run | Setup | Result |
|---|---|---|
| Single stratified 80/20 split (15% eval) | 6,290 train / 1,111 eval, early stopping, F1-tuned threshold | acc 0.9118, F1 0.9494 at threshold 0.20  Eoptimistic (threshold tuned on eval split) |
| Single stratified 80/20 split (20% eval) | 5,920 train / 1,481 eval, threshold 0.5, best epoch 7 | acc 0.9163, P 0.9378, R 0.9660, F1 0.9517 |
| **5-fold CV (current)** | fixed 7 epochs, threshold 0.5, vocab per fold | acc 0.8964 ± 0.0077, P 0.9519 ± 0.0109, R 0.9256 ± 0.0145, **F1 0.9384 ± 0.0049**; pooled TN 785 / FP 297 / FN 470 / TP 5,849 |
| **5-fold CV + GloVe 100d** | same split/seeds/epochs, fine-tuned embeddings | acc 0.9066 ± 0.0135, P 0.9560 ± 0.0076, R 0.9337 ± 0.0179, **F1 0.9446 ± 0.0085**; pooled TN 810 / FP 272 / FN 419 / TP 5,900. GloVe coverage 93.4 E3.6% of vocab |
| Pooled OOF threshold sweep (GloVe run) | 0.10 E.90 step 0.01 on out-of-fold probabilities | best thr 0.12: acc 0.9158, P 0.9318, R 0.9726, F1 0.9518 (mildly optimistic  Ethreshold chosen on the same predictions). F1 is flat from 0.10 to ~0.45, so 0.5 costs little |
| **5-fold CV + bert-base-uncased (stock)** | lr 2e-5, 3 epochs, batch 32, max_len 128, 10% warmup, wd 0.01, fp16, GPU 4060; tokenizer truncation only, no cleaning | acc 0.9550 ± 0.0023, P 0.9714 ± 0.0033, R 0.9761 ± 0.0032, **F1 0.9737 ± 0.0014**; pooled TN 900 / FP 182 / FN 151 / TP 6,168. Best OOF threshold 0.23 gives F1 0.9741 (negligible vs 0.5). ~2 min/fold |
| **5-fold CV + microsoft/deberta-v3-base (stock)** | identical harness/recipe; model cast to fp32 because the released checkpoint is fp16 (AMP unscale error otherwise); needs sentencepiece + protobuf | acc 0.9677 ± 0.0044, P 0.9783 ± 0.0040, R 0.9840 ± 0.0045, **F1 0.9811 ± 0.0026**; pooled TN 944 / FP 138 / FN 101 / TP 6,218. Best OOF threshold 0.60 gives F1 0.9815 (negligible vs 0.5). ~5 E min/fold (slower than BERT on this GPU) |

Interpretation: positive class reliable (~95% precision, 92.6% recall); negatives only 72.6% recall because `pos_weight` shifts the boundary toward positive. Fold spread ±0.005 means the score is stable.

## Why pretrained representations are used (justification)
A purely local model was trained first, with randomly initialised 128-d embeddings:
- Only 7,401 labelled reviews exist (5,920 per CV fold) for a ~1.16M-parameter network  Eroughly five training words per parameter.
- 46.7% of word types occur exactly once; 2.7% of held-out tokens are unseen. Vectors for rare words are therefore estimated from almost no evidence.
- Result was usable but limited: accuracy 0.8964, F1 0.9384, and the learning curves plateaued by epoch 5 E.
- Swapping only the embedding initialisation for GloVe 6B 100d (93.5% vocab coverage, fine-tuned) raised accuracy to 0.9066 and F1 to 0.9446 on identical folds, seeds, epochs and architecture.

Conclusion: the bottleneck is data volume, not the architecture or training recipe; local training alone cannot supply reliable representations for rare words. Pretrained vectors inject knowledge learned from billions of external tokens into the same model, and fine-tuning still uses only the training folds. The brief explicitly permits this ("pretrained word embeddings", "pretrained language models such as BERT", step 2/3), so BERT remains the next step if more accuracy is wanted.

## Options considered and rejected/deferred
| Option | Status | Reason |
|---|---|---|
| Logistic regression / TF-IDF baseline | rejected | User requires neural network; also barred as the solution |
| CNN text classifier | rejected | User requires one model only |
| Text cleaning (HTML/repeat chars/emoticons/spellcheck) | skipped | Audit shows text is already clean; only ~0.0 E.1 F1 available, and source truncation cannot be repaired |
| Separate validation split + early stopping | rejected for CV | Would tune on held-out fold data |
| Threshold tuning | deferred | Needs a clean 3-way split; otherwise optimistic |
| Pretrained GloVe embeddings | next candidate | Standard +1 E F1 on this dataset; requires one download |
| DistilBERT fine-tuning | deferred | 94 E6% possible but heavy; needs GPU install |
| bert-base-uncased stock fine-tune | tested (separate script) | F1 0.9737 ± 0.0014, acc 0.9550, ~10 min for 5 folds on GPU; `finetune_cv.py` |

## Current state
- `bilstm_cv.py`: BiLSTM 5-fold CV, reads `train.json`, writes `cv_results.png` and `threshold_sweep.png`. GloVe 100d loaded from `glove.6B.100d.txt` (347 MB, downloaded from downloads.cs.stanford.edu).
- `finetune_cv.py`: generic HuggingFace fine-tuning harness (any `AutoModelForSequenceClassification` checkpoint), same 5-fold split/seed, reads `train.json`, output plot name via `--out`. BERT run: `python finetune_cv.py`.
- `deberta_cv.py`: thin wrapper around `finetune_cv.py` with `--model-name microsoft/deberta-v3-base`; writes `deberta_results.png`.
- Logs: `results/cv_run.log` (BiLSTM), `results/bert_run.log` (BERT), `results/deberta_run.log` (DeBERTa).
- GPU now usable: torch 2.11.0+cu128, CUDA true on RTX 4060; transformers 5.20.2, sentencepiece 0.2.2, protobuf, tiktoken (DeBERTa tokenizer).
- Deleted stale artifacts: `cv_results_test.png` (savefig smoke test) and `model.pt` (BiLSTM checkpoint from the superseded 80% single-split run).
- `test.json` and `submission.csv` untouched.
