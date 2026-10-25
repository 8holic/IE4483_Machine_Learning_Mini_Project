# Decision log

Project: IE4483 Mini Project 1 — product-review sentiment classification (Amazon reviews).
Files under discussion: `train.json` (7,401 labeled), `test.json` (1,851 unlabeled), `submission.csv` (deliverable).

## Hard constraints (from user)
- One neural model only. No model mixing, no ensembles.
- `test.json` must not be read/used for training or evaluation.
- `submission.csv` must not be touched.
- No logistic regression / non-neural classifiers.
- Cross-validation over `train.json` is the agreed evaluation method.

## Environment
- `.venv` inside the project folder (user's choice).
- Python 3.14.6, torch 2.11.0+cu128 (GPU), scikit-learn 1.9.1, numpy 2.4.4, matplotlib 3.11.2,
  transformers 5.20.2, sentencepiece 0.2.2, protobuf, tiktoken.
- GPU: RTX 4060 (8 GB), CUDA available; CUDA torch replaced the original CPU-only build.

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
| `max_len=256` (BiLSTM), `128` (transformers) | BiLSTM p95 is 144 words; 256 is safe headroom. WordPiece/SentencePiece produce fewer tokens per word, so 128 covers the same text |
| `StratifiedKFold(5, shuffle=True, seed=42)` | Preserves the 85/15 class ratio per fold; every review tested once, trained on 4 times; fixed seed for reproducibility |
| Fixed epoch count for all folds, no early stopping | Avoids tuning on the held-out fold, which would make the CV score optimistic. BiLSTM 7 epochs; transformers 3 |
| Fresh model per fold, seeded init/shuffle | Independent folds, reproducible |
| BiLSTM: Embedding(100) → BiLSTM(128/direction) → masked mean+max pool → Dropout(0.4) → Linear(512→1) | Matches the assignment's "RNN + linear layer"; BiLSTM handles word order and contrast ("great, but broke"); ~1.16M params appropriate for 5.9k samples |
| Random-initialised embeddings | Superseded by GloVe 100d: random vectors learn little for the 47% of word types seen once |
| GloVe 6B 100d initialisation, fine-tuned during training | 93.5% of vocab covered; +0.6 F1 over random init for one download and no architecture change. Missing words stay random N(0, 0.1); `<pad>` row zeroed |
| Transformers: full fine-tuning of pretrained checkpoint, new 2-way head | Pretraining supplies language knowledge; fine-tuning still only sees training folds. Allowed by the brief (step 2/3) |
| Transformer recipe: lr 2e-5, 3 epochs, batch 32, AdamW wd 0.01, 10% linear warmup, grad clip 1.0, fp16 | Standard BERT-family fine-tuning; small LR preserves pretrained weights on 7.4k rows |
| DeBERTa checkpoint cast to fp32 | Released checkpoint is stored in fp16, which makes AMP unscale gradients fail |
| `BCEWithLogitsLoss` with `pos_weight ≈ 5.8` (BiLSTM) | 85/15 imbalance; without it, predicting all-positive is near-optimal. Transformers use plain cross-entropy (stock) |
| Decision threshold 0.5 per fold, plus a pooled out-of-fold sweep | 0.5 keeps folds comparable; the sweep shows where F1 peaks, flagged as mildly optimistic because it is picked on the same OOF predictions |
| Report accuracy, precision, recall, F1, confusion matrix, per-fold mean ± std | F1 is the honest headline under imbalance; std shows split stability |

## Experiments
| Run | Setup | Result |
|---|---|---|
| Single stratified 80/20 split (15% eval) | 6,290 train / 1,111 eval, early stopping, F1-tuned threshold | acc 0.9118, F1 0.9494 at threshold 0.20 — optimistic (threshold tuned on eval split) |
| Single stratified 80/20 split (20% eval) | 5,920 train / 1,481 eval, threshold 0.5, best epoch 7 | acc 0.9163, P 0.9378, R 0.9660, F1 0.9517 |
| **5-fold CV, random embeddings** | fixed 7 epochs, threshold 0.5, vocab per fold | acc 0.8964 ± 0.0077, P 0.9519, R 0.9256, **F1 0.9384 ± 0.0049**; pooled TN 785 / FP 297 / FN 470 / TP 5,849 |
| **5-fold CV + GloVe 100d (CPU)** | same split/seeds/epochs, fine-tuned embeddings | acc 0.9066 ± 0.0135, P 0.9560, R 0.9337, **F1 0.9446 ± 0.0085**; pooled TN 810 / FP 272 / FN 419 / TP 5,900. GloVe coverage 93.4–93.6% of vocab |
| **5-fold CV + GloVe 100d (GPU rerun)** | identical code/recipe, run on CUDA for a like-for-like device comparison | acc 0.8988 ± 0.0204, P 0.9570, R 0.9239, **F1 0.9394 ± 0.0141**; pooled TN 814 / FP 268 / FN 481 / TP 5,838. Fold 4 degraded (F1 0.9119, recall 0.8521) — GPU cuDNN numerics differ from CPU |
| Pooled OOF threshold sweep (BiLSTM, GPU) | 0.10–0.90 step 0.01 on out-of-fold probabilities | best thr 0.10: F1 0.9519, acc 0.9164, P 0.9347, R 0.9698 (mildly optimistic). F1 flat from 0.10 to ~0.45 |
| **5-fold CV + bert-base-uncased (stock)** | lr 2e-5, 3 epochs, batch 32, max_len 128, 10% warmup, wd 0.01, fp16, GPU 4060 | acc 0.9550 ± 0.0023, P 0.9714, R 0.9761, **F1 0.9737 ± 0.0014**; pooled TN 900 / FP 182 / FN 151 / TP 6,168. Best OOF threshold 0.23 gives F1 0.9741 (negligible vs 0.5). ~2 min/fold |
| **5-fold CV + google/electra-base-discriminator (RTD objective, BERT architecture)** | same harness/recipe/tokenizer (WordPiece 30k) as BERT; isolates the pretraining objective | acc 0.9623 ± 0.0037, P 0.9767, R 0.9793, **F1 0.9779 ± 0.0022**; pooled TN 934 / FP 148 / FN 131 / TP 6,188. Best OOF threshold 0.11 gives F1 0.9786. ~2 min/fold |
| **5-fold CV + microsoft/deberta-v3-base (stock)** | identical harness/recipe; fp32; needs sentencepiece + protobuf | acc 0.9677 ± 0.0044, P 0.9783, R 0.9840, **F1 0.9811 ± 0.0026**; pooled TN 944 / FP 138 / FN 101 / TP 6,218. Best OOF threshold 0.60 gives F1 0.9815 (negligible vs 0.5). ~5–6 min/fold |

Interpretation: positive class reliable (precision ~98% for DeBERTa); negatives remain the harder minority. DeBERTa raises negative recall from 75% (BiLSTM) to 87% (944/1,082) and cuts total errors from 751 to 239.

Paired per-fold comparison (same folds and seeds, one run per fold):
- DeBERTa > BERT on 5/5 folds, mean +0.0075 F1, paired t = 4.89, p = 0.008.
- DeBERTa > ELECTRA on 5/5 folds, mean +0.0032, p = 0.002.
- ELECTRA > BERT on 5/5 folds, mean +0.0043, p = 0.024.
So the ordering is consistent across every fold, not a single lucky split; residual caveat is one training run per fold (no multi-seed averaging).

## Why pretrained representations are used (justification)
A purely local model was trained first, with randomly initialised 128-d embeddings:
- Only 7,401 labelled reviews exist (5,920 per CV fold) for a ~1.16M-parameter network — roughly five training words per parameter.
- 46.7% of word types occur exactly once; 2.7% of held-out tokens are unseen. Vectors for rare words are therefore estimated from almost no evidence.
- Result was usable but limited: accuracy 0.8964, F1 0.9384, and the learning curves plateaued by epoch 5–6.
- Swapping only the embedding initialisation for GloVe 6B 100d (93.5% vocab coverage, fine-tuned) raised accuracy to 0.9066 and F1 to 0.9446 on identical folds, seeds, epochs and architecture.

Conclusion: the bottleneck is data volume, not the architecture or training recipe; local training alone cannot supply reliable representations for rare words. Pretrained knowledge (GloVe vectors, or a whole pretrained transformer) compensates, while fine-tuning still uses only the training folds. The brief explicitly permits this ("pretrained word embeddings", "pretrained language models such as BERT", step 2/3).

## Options considered and rejected/deferred
| Option | Status | Reason |
|---|---|---|
| Logistic regression / TF-IDF baseline | rejected | User requires neural network; also barred as the solution |
| CNN text classifier | rejected | User requires one model only |
| Text cleaning (HTML/repeat chars/emoticons/spellcheck) | skipped | Audit shows text is already clean; only ~0.0–0.1 F1 available, and source truncation cannot be repaired |
| Separate validation split + early stopping | rejected for CV | Would tune on held-out fold data |
| Threshold tuning | deferred | Needs a clean 3-way split; otherwise optimistic |
| DistilBERT fine-tuning | deferred | ~1 point below BERT at ~1.6× speed; possible wrapper, not needed |
| bert-base-uncased stock fine-tune | tested | F1 0.9737 ± 0.0014, acc 0.9550, ~10 min for 5 folds on GPU |
| microsoft/deberta-v3-base stock fine-tune | tested | F1 0.9811 ± 0.0026, acc 0.9677, ~30 min for 5 folds on GPU |
| google/electra-base-discriminator (RTD) | tested | F1 0.9779 ± 0.0022, acc 0.9623 — isolates the replaced-token-detection objective at BERT's architecture and tokenizer |

## Current state
- `models/bilstm_cv.py`: BiLSTM 5-fold CV, reads `train.json`, writes `results/bilstm_results.png` and `results/threshold_sweep.png`.
- `models/finetune_cv.py`: generic HuggingFace fine-tuning harness (any `AutoModelForSequenceClassification` checkpoint), same 5-fold split/seed, output plot name via `--out`. BERT run: `python models/finetune_cv.py`.
- `models/deberta_cv.py`: thin wrapper around `finetune_cv.py` with `--model-name microsoft/deberta-v3-base`; writes `results/deberta_results.png`.
- `models/electra_cv.py`: thin wrapper with `--model-name google/electra-base-discriminator`; writes `results/electra_results.png`.
- Logs: `results/bilstm_run.log`, `results/bert_run.log`, `results/deberta_run.log`, `results/electra_run.log`.
- Plots: `results/bilstm_results.png`, `results/RTD-bert_results.png`, `results/ELECTRA-deberta_results.png`, `results/electra_results.png`, `results/threshold_sweep.png`, `results/deberta_confusion_by_fold.png`.
- Deleted stale artifacts: `cv_results_test.png`, `model.pt`, old 4-panel `cv_results.png`, old CPU `cv_run.log`.
- `test.json` and `submission.csv` untouched.
