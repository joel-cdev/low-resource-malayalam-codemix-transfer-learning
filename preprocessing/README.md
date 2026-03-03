# Preprocessing Pipeline — Malayalam–English Code-Mixed Sentiment Analysis

## Overview

This notebook implements the full preprocessing pipeline for the FIRE 2020 Dravidian CodeMix dataset of Malayalam–English code-mixed YouTube comments. It produces cleaned CSV splits, a label map, class weights, and a code-mixing intensity feature — all consumed downstream by the fine-tuning and LoRA notebooks.

---

## Dataset

**Source:** FIRE 2020 – Dravidian CodeMix
**Task:** 5-class sentiment classification on Malayalam–English code-mixed text

| File | Format | Raw Rows | After Cleaning |
|------|--------|----------|----------------|
| `malayalam_train.tsv` | TSV | 4,851 | 4,840 |
| `malayalam_dev.tsv` | TSV | 540 | 540 |
| `malayalam_test_results.xlsx` | XLSX | 1,348 | 1,347 |

Each record contains two fields:
- `text` — the raw YouTube comment (may contain Malayalam script, romanised Malayalam, English, hashtags, and emoji)
- `category` — one of five sentiment labels

---

## Sentiment Classes

| Label | ID | Train Count | Train % |
|-------|----|-------------|---------|
| Positive | 0 | 2,018 | 41.7% |
| unknown_state | 1 | 1,340 | 27.7% |
| not-malayalam | 2 | 646 | 13.3% |
| Negative | 3 | 548 | 11.3% |
| Mixed_feelings | 4 | 288 | 6.0% |

> **Note for fine-tuning:** Only Positive (0) and Negative (3) are retained when training binary classification models. The remaining classes are excluded to reduce label noise.

---

## Pipeline Steps

### 1. Environment Setup & Imports
Installs `pandas` and `openpyxl`, then imports `re` and `json`.

### 2. Data Loading & Initial Inspection
Loads train and dev as tab-separated TSV files and test as an XLSX file. Verifies shapes and column names for consistency across splits.

### 3. Column Selection, Null Removal & Deduplication
- Retains only the `text` and `category` columns
- Drops rows with any null values
- Drops exact duplicate rows to prevent data leakage

### 4. Text Cleaning
A `clean_text` function is applied to all three splits:

| Operation | Rule |
|-----------|------|
| Whitespace normalisation | Collapses multiple spaces into one |
| URL removal | Strips `http://`, `https://`, and `www.` links |
| Mention removal | Strips `@username` tokens |
| Repeated punctuation capping | `!!!` → `!!`, `???` → `??` |
| Malayalam script | **Preserved as-is** — no transliteration or removal |

### 5. Label Standardisation & Encoding
Category strings are stripped of leading/trailing whitespace, then mapped to integer IDs via `label_map`:

```json
{"Positive": 0, "unknown_state": 1, "not-malayalam": 2, "Negative": 3, "Mixed_feelings": 4}
```

A NaN check confirms that all rows received a valid label.

### 6. Class Distribution Summary
Absolute counts and percentage shares are computed for the training split to document class imbalance.

### 7. Intermediate Save & Sanity Check
Cleaned DataFrames and the label map are saved locally before feature engineering:
- `clean_train.csv`, `clean_dev.csv`, `clean_test.csv`
- `label_map.json`

### 8. Code-Mix Ratio Analysis

An `english_word_ratio` feature is computed for each sample — the proportion of whitespace-delimited tokens that consist solely of ASCII letters (`[A-Za-z]+`). This serves as a proxy for code-mixing intensity.

**Mean English-word ratio by class (training set):**

| Category | Mean English Ratio |
|----------|--------------------|
| Mixed_feelings | 0.892 |
| Negative | 0.886 |
| unknown_state | 0.871 |
| Positive | 0.867 |
| not-malayalam | 0.823 |

Samples are also bucketed into a `codemix_bucket` column:

| Bucket | Criterion | Train Count |
|--------|-----------|-------------|
| `high_codemix` | 0.20 ≤ english_word_ratio ≤ 0.80 | 1,294 |
| `low_codemix` | ratio < 0.20 or ratio > 0.80 | 3,546 |

This column is used downstream to evaluate model robustness across varying degrees of code-mixing.

### 9. Class Weight Computation
Balanced class weights are computed with `sklearn.utils.class_weight.compute_class_weight` and saved to `class_weights.json` so both the full fine-tuning and LoRA runs use identical weighting.

| Label ID | Class | Weight |
|----------|-------|--------|
| 0 | Positive | 0.4797 |
| 1 | unknown_state | 0.7224 |
| 2 | not-malayalam | 1.4985 |
| 3 | Negative | 1.7664 |
| 4 | Mixed_feelings | 3.3611 |

Higher weights upweight minority classes (Negative, Mixed_feelings) during loss computation.

### 10. Final Export to Google Drive
All artefacts are copied to `MyDrive/Project2_preprocessing_mal_en/` for use in downstream Colab notebooks.

---

## Output Artefacts

| File | Description |
|------|-------------|
| `clean_train.csv` | Cleaned training split (4,840 rows): `text`, `category`, `label_id` |
| `clean_dev.csv` | Cleaned validation split (540 rows) |
| `clean_test.csv` | Cleaned test split (1,347 rows) |
| `label_map.json` | String → integer label mapping |
| `class_weights.json` | Integer label ID → balanced class weight |

Each CSV also includes `english_word_ratio` and `codemix_bucket` columns added in Step 8.

---

## File Structure

```
preprocessing/
├── Project2_preprocessing_mal_en.ipynb   # This notebook
├── clean_train.csv                        # Cleaned training data
├── clean_dev.csv                          # Cleaned validation data
├── clean_test.csv                         # Cleaned test data
├── label_map.json                         # Label encoding
├── class_weights.json                     # Balanced class weights
└── README.md                              # This file
```

---

## How to Run

1. Open `Project2_preprocessing_mal_en.ipynb` in **Google Colab**
2. Upload or mount the three raw data files:
   - `malayalam_train.tsv`
   - `malayalam_dev.tsv`
   - `malayalam_test_results.xlsx`
3. Mount Google Drive when prompted
4. Run all cells in order
5. Artefacts are saved to both the local Colab session and `MyDrive/Project2_preprocessing_mal_en/`

---

## Requirements

```
pandas
openpyxl
scikit-learn
numpy
```

All are pre-installed in Google Colab.
