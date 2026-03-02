****Low-Resource Malayalam–English Code-Mixed Transfer Learning****

This repository contains the full implementation of our transfer learning pipeline for low-resource Malayalam–English code-mixed sentiment analysis.

The project investigates how multilingual Large Language Models (LLMs) can be adapted to linguistically complex, code-mixed data using:

Full Fine-Tuning

Parameter-Efficient Fine-Tuning (LoRA)

**Dataset**

Malayalam–English code-mixed YouTube comments (FIRE 2020 – Dravidian CodeMix).

Original dataset includes five sentiment classes:

Positive

Negative

Mixed_feelings

unknown_state

not-malayalam

Original train/dev/test splits were preserved.

For fine-tuning comparison, the task was reformulated as binary sentiment classification (Positive vs Negative) to enable controlled evaluation between full fine-tuning and LoRA.

Ambiguous or non-polar classes (unknown_state, not-malayalam, Mixed_feelings) were excluded during training.**

**Project Components**

**1. Data & Preprocessing**

Removed null rows and duplicates

Minimal normalization (URLs, mentions, whitespace)

Preserved code-mixing and Malayalam script

Retained original 5-class sentiment structure

Encoded labels to numeric IDs

Computed English-token ratio as proxy for code-mixing intensity

Computed balanced class weights for handling class imbalance

Artifacts:

preprocessing/Project2_preprocessing_mal_en.ipynb

preprocessing/label_map.json

preprocessing/class_weights.json

**2. Model Design & Full Fine-Tuning**

Base multilingual model selection (mT5)

Binary sentiment classification setup

Fine-tuning using HuggingFace Transformers

Files:

finetuning/02_finetuning_csci316_project2_FIXED.ipynb

**3. PEFT / LoRA**

LoRA-based parameter-efficient fine-tuning

Reduced trainable parameter comparison

Efficiency analysis

(To be expanded)

**4. Evaluation & Analysis**

Accuracy

Macro-F1 score

Confusion Matrix

Code-mix robustness analysis

(To be expanded)

**5. Deployment**

Docker containerization

Inference API

Reproducibility instructions

(To be expanded)

**Trained Model Weights**

Due to GitHub file size limitations, the trained model weights (model.safetensors, ~1.12GB) are stored in the shared Google Drive folder:

Google Drive Link:
https://drive.google.com/drive/folders/1VnzX-YACtcRdCc6vWtc99b9VVEs0V95W?usp=sharing
