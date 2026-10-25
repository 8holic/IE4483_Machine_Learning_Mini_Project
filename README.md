# IE4483 Project 1  Eproduct-review sentiment classification

Run every command from this project root; all data paths are relative to it.

## Layout

```
train.json              labelled reviews (7,401)
test.json               unlabelled reviews (1,851)  Enever read by the scripts
submission.csv          deliverable  Euntouched
glove.6B.100d.txt       GloVe vectors (347 MB) used by the BiLSTM
.venv\                  Python 3.14 environment

models/
  bilstm_cv.py           BiLSTM + GloVe, 5-fold CV
  finetune_cv.py        generic HuggingFace fine-tuning harness, 5-fold CV (BERT by default)
  deberta_cv.py         same harness, DeBERTa-v3-base

results/                plots and run logs from all experiments
notes/decision_log.md   every decision, its reason, and every experiment result
reference/              assignment PDF and the provided TF-IDF notebook
```

## Setup

Not committed to git: the virtualenv and the 347 MB GloVe file.

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
curl -o glove.6B.zip https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
Expand-Archive glove.6B.zip -DestinationPath .
# keep glove.6B.100d.txt in the project root
```

For GPU, install the CUDA build instead of the default wheel:

```
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

## Commands

```
.\.venv\Scripts\python.exe models\bilstm_cv.py      # BiLSTM + GloVe
.\.venv\Scripts\python.exe models\finetune_cv.py   # BERT-base
.\.venv\Scripts\python.exe models\deberta_cv.py    # DeBERTa-v3-base
```

Any other compatible checkpoint:

```
.\.venv\Scripts\python.exe models\finetune_cv.py --model-name roberta-large --out results/roberta_results.png
```

Every run only reads `train.json` and writes to `results/`. `test.json` and `submission.csv` are not touched without explicit instruction.

## Current results (5-fold CV)

| Model | Accuracy | F1 |
|---|---|---|
| BiLSTM + GloVe | 0.9066 | 0.9446 |
| BERT-base | 0.9550 | 0.9737 |
| DeBERTa-v3-base | 0.9677 | 0.9811 |
