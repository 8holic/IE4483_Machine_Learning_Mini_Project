# Feature format and preprocessing

Answers the brief's question (a): choice of feature format, and how raw review text is converted
into model input. Every claim below is measured on `train.json` (see `decision_log.md` for runs).

## 1. Chosen feature format

**Pretrained contextual token representations** (fine-tuned transformer):
`bert-base-uncased` and `microsoft/deberta-v3-base`, selecting the `[CLS]` representation through a
2-way linear head. For the local-model comparison, **pretrained static word vectors** (GloVe 6B 100d)
initialise the BiLSTM's embedding layer.

Three candidate formats were considered:

| Format | Used? | Why / why not |
|---|---|---|
| TF-IDF / bag-of-words | No | Loses word order and negation scope (`not good` vs `good not`); would also be a non-neural classifier, which the task disallows |
| Static word embeddings (GloVe 100d) + BiLSTM | Yes (baseline) | Order-aware and neural; 93.5% of the vocabulary has a pretrained vector. Remaining words are learned from only 5,920 reviews (46.7% of word types occur once), so rare-word vectors are weak |
| Pretrained language model (BERT / DeBERTa) + linear head | Yes (main) | Contextual vectors (one per word occurrence, not per word type), subword tokenisation, and language knowledge from billions of external tokens. Highest accuracy in CV: BERT F1 0.9737, DeBERTa F1 0.9811 vs BiLSTM 0.9446 |

The pretraining corpora are external (Wikipedia/books for BERT; DeBERTa's own corpus). No held-out
review is used for pretraining, so there is no label leakage.

## 2. Preprocessing for the BiLSTM (word-level)

1. **Tokenise**: lowercase, regex `[a-z0-9']+`. Keeps apostrophes so negations (`don't`, `isn't`)
   survive; punctuation is dropped (the audit found emoticons in only 39 reviews, so nothing is lost
   in practice).
2. **Vocabulary, rebuilt per CV fold from that fold's training 80% only**:
   - count tokens, discard types seen fewer than 2 times (`min_freq=2`) — singletons are mostly
     typos and cannot be learned from one occurrence;
   - keep the 30k most frequent (actual size ~6,930 per fold);
   - reserve ID 0 for `<pad>` and 1 for `<unk>`.
3. **Encode**: map tokens to IDs, truncate to 256 tokens. Measured review length: median 42, p95 144,
   so only 1.18% of reviews are cut.
4. **Embedding initialisation**: 100-d GloVe vector for every vocabulary word GloVe knows
   (93.4–93.6% coverage), missing words random N(0, 0.1), `<pad>` row fixed at zero. Vectors are
   then fine-tuned during training.
5. **Batching**: dynamic padding only to the longest review in each batch (64 reviews), producing a
   `[batch, length]` integer tensor.

## 3. Preprocessing for BERT / DeBERTa (subword-level)

1. **No manual cleaning.** An audit of all 7,401 reviews found 0 HTML tags/entities, 2 URLs, 18
   non-ASCII reviews, and only 0.03% elongated tokens. Cleaning would change nothing measurable, and
   source-truncated words (e.g. "...good pric") cannot be repaired.
2. **Tokenizer-native normalisation**: `AutoTokenizer` applies the checkpoint's own rules
   (WordPiece + lowercasing for `bert-base-uncased`; SentencePiece for DeBERTa-v3, 128k vocabulary).
   Rare words and misspellings become subword pieces instead of a single `<unk>`.
3. **Max length 128 subword tokens**, truncation enabled; special tokens (`[CLS]`/`<s>`) added by the
   tokenizer. Subword tokenisation packs the same text into fewer units than words.
4. **Dynamic padding** per batch via the tokenizer's data collator; attention mask marks real tokens
   so padding never enters self-attention.
5. **Labels**: 0/1 directly; no label engineering.

Shared by all models: stratified 5-fold split (seed 42) so the 85.4/14.6 class ratio is preserved,
and the vocabulary/tokenizer statistics of a fold never include that fold's held-out reviews.

## 4. Differences in how the models are trained

| | BiLSTM + GloVe | BERT / DeBERTa |
|---|---|---|
| Starting point | random BiLSTM weights; GloVe-initialised embedding | pretrained transformer weights; new random 2-way head |
| Loss | BCEWithLogitsLoss, `pos_weight ≈ 5.8` | cross-entropy (stock, no class weighting) |
| Optimiser / lr | Adam, 1e-3 | AdamW, 2e-5, weight decay 0.01 |
| Schedule | none | 10% linear warmup, then linear decay |
| Epochs | 7 (fixed) | 3 (fixed) |
| Batch | 64 | 32 |
| Regularisation | dropout 0.4, grad clip 1.0 | grad clip 1.0, weight decay, fp16 AMP |
| Max input | 256 words | 128 subword tokens |
| Why different | trained from scratch on 5.9k examples needs a larger LR and more epochs; explicit class weighting compensates the 85/15 split | pretrained weights need a small LR to avoid catastrophic forgetting; the model already handles imbalance better, so stock CE is kept |
| Extra requirement | — | DeBERTa checkpoint is stored in fp16, so it is cast to fp32 before AMP |

Both pipelines use the same folds, seed, threshold (0.5) and evaluation code, so the comparison is
like-for-like.

## 5. Why this format wins (evidence)

- Random embeddings → GloVe, same architecture/folds: F1 0.9384 → 0.9446. Only the embedding
  initialisation changed, isolating the value of pretrained word vectors.
- BiLSTM → BERT, same folds: F1 0.9446 → 0.9737; missed positives fall 419 → 151.
- BERT → DeBERTa-v3, same folds/recipe: F1 0.9737 → 0.9811, errors 333 → 239, driven by negative
  recall rising from 83.2% to 87.2%. Mechanisms: disentangled attention (relative position for
  negation/contrast), replaced-token-detection pretraining (denser signal), and a 128k SentencePiece
  vocabulary that reduces out-of-vocabulary loss.
- BERT → ELECTRA-base (same 12-layer BERT architecture and WordPiece 30k tokenizer, only the
  pretraining objective changes to replaced-token detection): F1 0.9737 → 0.9779. This isolates the
  objective: ~+0.4 F1 of DeBERTa's advantage comes from RTD alone; the rest comes from DeBERTa's
  disentangled attention and larger SentencePiece vocabulary.

Ceiling note: human labels on this data contain ambiguous, sarcastic and mixed reviews, so ~2–3%
irreducible error remains regardless of feature format; perfect test accuracy is not a meaningful
target.
