# Low-Resource Malayalam–English Code-Mixed Transfer Learning

This project evaluates transfer learning strategies for low-resource Malayalam–English code-mixed sentiment classification using multilingual transformer models.

The repository contains the full implementation of our pipeline exploring:

- Full Fine-Tuning
- Parameter-Efficient Fine-Tuning (LoRA)

---

## Dataset

Malayalam–English code-mixed YouTube comments (FIRE 2020 – Dravidian CodeMix).

The original dataset includes five sentiment classes:

- Positive
- Negative
- Mixed_feelings
- unknown_state
- not-malayalam

The original train/dev/test splits were preserved.

For controlled comparison between full fine-tuning and LoRA, the task was reformulated as **binary sentiment classification (Positive vs Negative)**.

Ambiguous or non-polar classes (`unknown_state`, `not-malayalam`, `Mixed_feelings`) were excluded during model training to reduce label noise and improve interpretability of evaluation metrics.

---

## Project Components

### 1. Data & Preprocessing

- Removed null rows and duplicates
- Applied minimal normalization (URLs, mentions, whitespace)
- Preserved code-mixing and Malayalam script
- Retained original 5-class structure
- Encoded labels to numeric IDs
- Computed English-token ratio as proxy for code-mixing intensity
- Computed balanced class weights to address class imbalance

**Artifacts:**

- `preprocessing/Project2_preprocessing_mal_en.ipynb`
- `preprocessing/label_map.json`
- `preprocessing/class_weights.json`

---

### 2. Model Design & Full Fine-Tuning

- Base multilingual model selection (mT5)
- Binary sentiment classification setup
- Fine-tuning using HuggingFace Transformers

**File:**

- `finetuning/02_finetuning_V1.ipynb`

---

### 3. PEFT / LoRA

- LoRA-based parameter-efficient fine-tuning
- Reduced trainable parameter comparison
- Training efficiency analysis

*(To be expanded)*

---

### 4. Evaluation & Analysis

- Accuracy
- Macro-F1 score
- Confusion Matrix
- Code-mix robustness evaluation

*(To be expanded)*

---

### 5. Deployment

- Docker containerization
- Inference API
- Reproducibility instructions

*(To be expanded)*

---

## Trained Model Weights

Due to GitHub file size limitations, the trained model weights (`model.safetensors`, ~1.12GB) are stored in the shared Google Drive folder:

**Google Drive Link:**  
https://drive.google.com/drive/folders/1VnzX-YACtcRdCc6vWtc99b9VVEs0V95W?usp=sharing

To load the trained model after downloading the `trained_model_mt5` folder:

```python
from transformers import MT5ForConditionalGeneration

model = MT5ForConditionalGeneration.from_pretrained("path_to_trained_model_folder")
