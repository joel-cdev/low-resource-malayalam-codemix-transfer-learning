# CSCI316 Project 2 — Multilingual Sentiment Analysis with mT5

## Overview
This project fine-tunes a multilingual transformer model (mT5-small) to perform **binary sentiment analysis** (Positive / Negative) on a Malayalam-English dataset. The model learns to classify text as either positive or negative sentiment, handling code-switched and multilingual input.

---

## Dataset
The dataset consists of Malayalam/English social media text, pre-cleaned and split into three files stored in Google Drive:

| File | Split | Size (after filtering) |
|------|-------|------------------------|
| `clean_train.csv` | Training | 2,566 examples |
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

## Project Pipeline

### 1. Data Loading & Filtering
- Load train/dev/test CSVs from Google Drive
- Filter to keep only Positive and Negative categories

### 2. Format Conversion
- Convert each row into input/output pairs:
  - Input: `"sentiment: <text>"`
  - Target: `"positive"` or `"negative"`

### 3. Tokenization
- Tokenize inputs (max length 64 tokens)
- Tokenize labels (max length 8 tokens)
- Replace padding tokens in labels with `-100` so they are ignored during loss calculation

### 4. Training
- Fine-tune mT5-small using HuggingFace `Seq2SeqTrainer`
- Evaluate after every epoch on the dev set
- Save the best model checkpoint

### 5. Evaluation
- Evaluate on dev set and test set
- Metrics: Accuracy and Weighted F1 Score
- Decode generated outputs and normalize before comparing to gold labels

### 6. Saving
- Model and tokenizer saved to Google Drive: `MyDrive/mt5_finetuned/`

---

## Training Configuration

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
| FP16 (mixed precision) | Yes (GPU) |

---

## Results

### Training Progress
| Epoch | Training Loss | Validation Loss | Accuracy | F1 Weighted |
|-------|--------------|-----------------|----------|-------------|
| 1 | 2.9700 | 1.1303 | 0.8145 | 0.7313 |
| 2 | 1.0073 | 0.5275 | 0.8145 | 0.7313 |
| 3 | 0.6552 | 0.3519 | 0.8145 | 0.7313 |

### Final Scores

| Split | Accuracy | F1 Weighted |
|-------|----------|-------------|
| Dev | **81.45%** | **73.13%** |
| Test | **80.34%** | **71.58%** |

---

## Key Observations
- Training loss dropped significantly from **2.97 → 0.65** across 3 epochs, showing the model learned effectively
- Accuracy stabilized after epoch 1, suggesting the model converged quickly on this dataset
- The model shows a slight bias toward predicting "positive" — likely due to class imbalance in the training data
- The gap between Accuracy (81%) and F1 (73%) reflects this imbalance

---

## File Structure
```
Project2_preprocessing_mal_en/
├── clean_train.csv          # Training data
├── clean_dev.csv            # Validation data
├── clean_test.csv           # Test data
├── class_weights.json       # Class weights
├── label_map.json           # Label mapping
└── mt5_finetuned/           # Saved fine-tuned model
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── spiece.model
    └── special_tokens_map.json

Notebook:
└── 02_finetuning_csci316_project2_FIXED.ipynb
```

---

## How to Run

1. Open the notebook in Google Colab with a GPU runtime (T4 or better)
2. Mount Google Drive when prompted
3. Ensure your CSV files are in `MyDrive/Project2_preprocessing_mal_en/`
4. Run all cells in order
5. Training takes approximately **1.5 hours** on a T4 GPU for 3 epochs

---

## Requirements
```
transformers
datasets
scikit-learn
torch
pandas
numpy
```
All are pre-installed in Google Colab.

---

## Notes
- The `pin_memory` warning that appears during training is harmless and can be ignored
- If Google Drive shows as read-only, re-run the `drive.mount()` cell and try again
- The model is saved locally to `/content/mt5_finetuned/` first, then copied to Drive