Low-Resource Malayalam–English Code-Mixed Transfer Learning

This repository contains the full implementation of our transfer learning pipeline for low-resource Malayalam–English code-mixed sentiment analysis.

The project explores how multilingual Large Language Models (LLMs) can be adapted to under-represented, code-mixed language settings using both full fine-tuning and parameter-efficient fine-tuning (PEFT).

Project Components
1. Data & Preprocessing

Removed null rows and duplicates

Minimal normalization (URLs, mentions, whitespace)

Preserved code-mixing and Malayalam script

Retained original 5-class sentiment structure

Encoded labels to numeric IDs

Computed English-token ratio as proxy for code-mixing intensity

Computed balanced class weights for training

Artifacts:

Project2_preprocessing_mal_en.ipynb

label_map.json

class_weights.json

2. Model Design & Full Fine-Tuning

(To be added)

Base multilingual model selection

Full parameter fine-tuning

Cross-framework implementation

3. PEFT / LoRA

(To be added)

LoRA adaptation

Parameter comparison

Training efficiency analysis

4. Evaluation & Analysis

(To be added)

Standard metrics (Accuracy, Macro-F1)

Code-mix robustness evaluation

Failure analysis

5. Deployment

(To be added)

Docker containerization

Inference API

Reproducibility instructions

Dataset

Malayalam–English code-mixed YouTube comments (FIRE 2020 – Dravidian CodeMix).

Five sentiment classes:

Positive

Negative

Mixed_feelings

unknown_state

not-malayalam

Original train/dev/test splits were preserved.


