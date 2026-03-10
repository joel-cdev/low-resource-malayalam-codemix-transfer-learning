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

- `finetuning/03_full_finetuning_v2_the_final_baseline.ipynb`

---

### 3. PEFT / LoRA

- LoRA-based parameter-efficient fine-tuning
- Reduced trainable parameter comparison
- Training efficiency analysis

**Files:**

- `lora/05_lora_v2_final.ipynb` — final LoRA model
- `lora/05_lora_v2_cmrs_evaluation.ipynb` — final LoRA + CMRS evaluation

---

### 4. Evaluation & Analysis

Both models were evaluated on the same held-out test set (702 examples) using accuracy, weighted F1, and macro F1.

| Model | Test Accuracy | Weighted F1 | Macro F1 |
|-------|--------------|-------------|----------|
| Full Fine-Tuning (V2) | **84.62%** | **81.12%** | **65.6%** |
| LoRA (V2) | 80.34% | 71.58% | 44.55% |

Full fine-tuning outperforms LoRA across all metrics, with the largest gap in macro F1 — reflecting that full fine-tuning handles the minority (negative) class better.

**Code-Mix Robustness Score (CMRS)** was computed to evaluate how sensitive the LoRA model is to varying levels of Malayalam–English code-mixing. The test set was divided into two subsets based on the English-word ratio, defined as the proportion of tokens written in the Latin alphabet within each comment.

Comments were grouped as follows:

- **Low code-mix:** English ratio < 30% (predominantly Malayalam text)
- **High code-mix:** English ratio ≥ 30% (significant English mixing)

The model’s weighted F1 score was then calculated separately for each subset:

| Subset | F1 |
|--------|----|
| Low code-mix (< 30% English) | 0.889 |
| High code-mix (≥ 30% English) | 0.682 |

The Code-Mix Robustness Score (CMRS) is defined as:

CMRS = 1 − |F1_high − F1_low|

Using the measured values:

CMRS = 1 − |0.889 − 0.682| ≈ 0.793

A CMRS of 0.79 indicates moderate robustness to code-mixing. The LoRA model performs substantially better on comments that are primarily Malayalam, while its performance decreases when the proportion of English words increases. This suggests that the parameter-efficient fine-tuning approach struggles to adapt fully to heavily code-mixed inputs, likely due to limited training data and restricted parameter updates in the LoRA adapters.

Overall, this result supports the earlier quantitative findings, where the LoRA model exhibited strong bias toward the majority class and reduced ability to capture complex linguistic patterns present in highly code-mixed social media text.

---

### 5. Deployment

The fine-tuned model is served via a FastAPI REST API containerised with Docker.

```bash
cd deployment
docker build -t malayalam-sentiment-api .
docker run -p 8000:8000 malayalam-sentiment-api
```

Swagger UI available at: http://localhost:8000/docs

See `deployment/README_deployment.md` for full setup instructions.

---

## Trained Model Weights

Due to GitHub file size limitations, the trained model weights (`model.safetensors`, ~1.12GB) are stored in the shared Google Drive folder:

**Google Drive Link:**  
https://drive.google.com/drive/folders/1VnzX-YACtcRdCc6vWtc99b9VVEs0V95W?usp=sharing

To load the trained model after downloading the `trained_model_mt5` folder:

```python
from transformers import MT5ForConditionalGeneration

model = MT5ForConditionalGeneration.from_pretrained("path_to_trained_model_folder")
```
