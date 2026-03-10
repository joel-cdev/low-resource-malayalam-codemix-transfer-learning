# CSCI316 Project 2 — Multilingual Sentiment Analysis with mT5

## Overview
This project applies parameter-efficient fine-tuning using LoRA to a multilingual transformer model (mT5-small) to perform binary sentiment analysis (Positive / Negative) on a Malayalam-English code-mixed dataset. The model learns to classify social media text as either positive or negative sentiment while handling code-switched and multilingual input.

Unlike the full fine-tuning baseline, this version uses LoRA adapters, which update only a small number of additional parameters while keeping the original model weights frozen. This reduces training cost while still allowing the model to adapt to the sentiment classification task.

The current version in use is the final LoRA implementation (03_lora_csci316_project2_malayalam_finalversion.ipynb).
An earlier experiment (03_lora_csci316_project2_malayalam_version4.ipynb) was used during development.

---

## Notebooks

| File | Description |
|------|-------------|
| `03_lora_csci316_project2_malayalam_finalversion.ipynb` | **Current version — LoRA implementation** |
| `03_lora_csci316_project2_malayalam_version4.ipynb` | Earlier LoRA experiment |


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

### 2. Class Balancing
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

### 5. Training (LoRA)
- mT5-small using HuggingFace `Seq2SeqTrainer`
- Evaluate after every epoch on the dev set
- Save the best model checkpoint (based on weighted F1)

### 6. Evaluation
- Evaluate on dev set and test set
- Metrics: Accuracy, Weighted F1, Macro F1, classification report, confusion matrix
- Decode generated outputs and normalize before comparing to gold labels

### 7. Saving
- The trained model (with LoRA adapters) and tokenizer are saved to Google Drive

---

## Training Configuration (V2)

| Hyperparameter | Value |
|----------------|-------|
| Base Model | google/mt5-small |
| Training Method | LoRA (PEFT adapters) |
| Learning Rate | 3e-4 |
| Batch Size | 16 |
| Epochs | 3 |
| Weight Decay | 0.01 |
| Warmup Steps | 100 |
| Max Input Length | 64 tokens |
| Max Label Length | 8 tokens |
| Evaluation Strategy | Every epoch |
| Best model selection | Max weighted F1 |
| FP16 | Disabled (training stability) |

---

## Results

### Training Progress
| Epoch | Training Loss | Validation Loss | Accuracy | F1 Weighted |
|-------|--------------|-----------------|----------|-------------|
| 1 | 7.5004 | 2.9451 | 0.1855 | 0.0580 |
| 2 | 3.0097 | 0.7736 | 0.8145 | 0.7313 |
| 3 | 2.1015 | 0.5141 | 0.8145 | 0.7313 |

Best checkpoint selected at **epoch 2**.

### Final Scores

| Split | Accuracy | F1 Weighted | F1 Macro |
|-------|----------|-------------|----------|
| Dev | **81.45%** | **73.13%** | **44.89%** |
| Test | **80.34%** | **71.58%** | **44.55%** |

---

## Key Observations
- LoRA reduces the number of trainable parameters compared to full fine-tuning
- Training remains fast while adapting the model to the sentiment classification task
- The model achieves reasonable accuracy but tends to predict **positive sentiment more often**
- This results in **lower macro F1**, indicating weaker performance on the minority negative class

---

## File Structure
```
lora/
├── 03_lora_csci316_project2_malayalam_finalversion.ipynb # Current version
├── 03_lora_csci316_project2_malayalam_version4.ipynb                 # Old experiment
└── README.md

Saved model (Google Drive):
└── mt5_finetuned/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── README.md
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── training_args.bin
```

---

## How to Run

1. Open `03_lora_csci316_project2_malayalam_finalversion.ipynb` in Google Colab with a GPU runtime (T4 or better)
2. Mount Google Drive when prompted
3. Ensure your CSV files are in `MyDrive/Project2_preprocessing_mal_en/`
4. Run all cells in order
5. Training takes approximately **5–6 minutes** on a T4 GPU for 3 epochs

---

## Requirements
```
transformers
datasets
peft
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
- The model is saved locally to `/content/mt5_finetuned/` first, then copied to Drive
