# CSCI316 Project 2 — Multilingual Sentiment Analysis with mT5

## Overview
This project fine-tunes a multilingual transformer model (mT5-small) to perform **binary sentiment analysis** (Positive / Negative) on a Malayalam-English code-mixed dataset. The model learns to classify text as either positive or negative sentiment, handling code-switched and multilingual input.

The current version in use is **V2** (`03_full_finetuning_V2_the_final_baseline.ipynb`), which is the final baseline. V1 (`02_finetuning_V1.ipynb`) was an earlier experiment and has since been superseded.

---

## Notebooks

| File | Description |
|------|-------------|
| `03_full_finetuning_V2_the_final_baseline.ipynb` | **Current version — final baseline** |
| `02_finetuning_V1.ipynb` | Old experiment — superseded by V2 |

---

## What Changed from V1 to V2

| Change | V1 | V2 |
|--------|----|----|
| Class balancing | None (raw imbalanced data: 2018 pos / 548 neg) | Oversampling minority class → 2018 pos / 2018 neg (4036 total) |
| FP16 | `fp16=torch.cuda.is_available()` | `fp16=False` (disabled for training stability) |
| GPU placement | Implicit | Explicit `model.to("cuda")` |
| Evaluation metrics | Accuracy + weighted F1 only | Accuracy, weighted F1, **macro F1**, classification report, confusion matrix |
| Saved model path | `mt5_finetuned/` | `mt5_finetuned_updated/` |
| Test Accuracy | 80.34% | **84.62%** |
| Test Weighted F1 | 71.58% | **81.12%** |

The most impactful change was **oversampling the negative class** to address class imbalance, which dramatically improved both accuracy and F1.

---

## Dataset
The dataset consists of Malayalam/English social media text (FIRE 2020 – Dravidian CodeMix), pre-cleaned and split into three files stored in Google Drive:

| File | Split | Size (after filtering) |
|------|-------|------------------------|
| `clean_train.csv` | Training | 2,566 examples (raw); 4,036 after oversampling |
| `clean_dev.csv` | Validation | 275 examples |
| `clean_test.csv` | Test | 702 examples |

Only **Positive** and **Negative** labels were kept (other categories were removed).

Each example has:
- `text` — the input sentence
- `category` — the sentiment label (Positive / Negative)

---

## Model
**Base model:** [`google/mt5-small`](https://huggingface.co/google/mt5-small)

mT5 (Multilingual T5) is a sequence-to-sequence transformer pre-trained on 101 languages. It is well suited for multilingual tasks like this one because it already understands both Malayalam and English from pre-training.

The task is framed as **text generation**: given the input `"sentiment: <text>"`, the model generates either `"positive"` or `"negative"`.

---

## Project Pipeline (V2)

### 1. Data Loading & Filtering
- Load train/dev/test CSVs from Google Drive
- Filter to keep only Positive and Negative categories

### 2. Class Balancing (new in V2)
- Group training examples by label
- Oversample the negative class to match the positive class count
- Result: balanced training set of 4,036 examples (2,018 per class)
- Dev and test sets are **not** oversampled

### 3. Format Conversion
- Convert each row into input/output pairs:
  - Input: `"sentiment: <text>"`
  - Target: `"positive"` or `"negative"`

### 4. Tokenization
- Tokenize inputs (max length 64 tokens)
- Tokenize labels (max length 8 tokens)
- Replace padding tokens in labels with `-100` so they are ignored during loss calculation

### 5. Training
- Fine-tune mT5-small using HuggingFace `Seq2SeqTrainer`
- Evaluate after every epoch on the dev set
- Save the best model checkpoint (based on weighted F1)

### 6. Evaluation
- Evaluate on dev set and test set
- Metrics: Accuracy, Weighted F1, Macro F1, classification report, confusion matrix
- Decode generated outputs and normalize before comparing to gold labels

### 7. Saving
- Model and tokenizer saved to Google Drive: `MyDrive/mt5_finetuned_updated/`

---

## Training Configuration (V2)

| Hyperparameter | Value |
|----------------|-------|
| Base Model | google/mt5-small |
| Learning Rate | 3e-4 |
| Batch Size | 16 |
| Epochs | 3 |
| Weight Decay | 0.01 |
| Warmup Steps | 100 |
| Max Input Length | 64 tokens |
| Max Label Length | 8 tokens |
| Evaluation Strategy | Every epoch |
| Best model selection | Max weighted F1 |
| FP16 (mixed precision) | No (disabled for stability) |

---

## Results (V2 — Final Baseline)

### Training Progress
| Epoch | Training Loss | Validation Loss | Accuracy | F1 Weighted |
|-------|--------------|-----------------|----------|-------------|
| 1 | ~1.57 | 0.2013 | 0.8145 | 0.7313 |
| 2 | ~0.52 | 0.1668 | 0.8655 | 0.8428 |
| 3 | ~0.37 | 0.1988 | 0.8255 | 0.8212 |

Best checkpoint selected at **epoch 2** (highest weighted F1).

### Final Scores

| Split | Accuracy | F1 Weighted | F1 Macro |
|-------|----------|-------------|----------|
| Dev | **86.55%** | **84.28%** | — |
| Test | **84.62%** | **81.12%** | **65.6%** |

---

## Key Observations
- Oversampling the minority (negative) class was the single biggest improvement, pushing test weighted F1 from ~72% to ~81%
- FP16 was disabled after observing instability; training on GPU without mixed precision remained fast enough
- Macro F1 (65.6%) vs Weighted F1 (81.1%) gap reflects the still-challenging minority class
- Best checkpoint at epoch 2 — epoch 3 slightly overfit on the balanced training set

---

## File Structure
```
finetuning/
├── 03_full_finetuning_V2_the_final_baseline.ipynb   # Current version
├── 02_finetuning_V1.ipynb                            # Old experiment
└── README.md

Saved model (Google Drive):
└── mt5_finetuned_updated/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── spiece.model
    └── special_tokens_map.json
```

---

## How to Run

1. Open `03_full_finetuning_V2_the_final_baseline.ipynb` in Google Colab with a GPU runtime (T4 or better)
2. Mount Google Drive when prompted
3. Ensure your CSV files are in `MyDrive/Project2_preprocessing_mal_en/`
4. Run all cells in order
5. Training takes approximately **7–8 minutes** on a T4 GPU for 3 epochs

---

## Requirements
```
transformers
datasets
scikit-learn
torch
pandas
numpy
matplotlib
```
All are pre-installed in Google Colab.

---

## Notes
- The `pin_memory` warning that appears during training is harmless and can be ignored
- If Google Drive shows as read-only, re-run the `drive.mount()` cell and try again
- The model is saved locally to `/content/mt5_finetuned_updated/` first, then copied to Drive
