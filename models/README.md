# Models

## mt5_finetuned_updated (Final Baseline)

The fine-tuned mT5-small model is **not stored in this repository** due to GitHub's file size limits (`model.safetensors` is ~300MB).

### How to get the model

**Option 1 — Re-run the fine-tuning notebook (recommended)**
- Open `finetuning/03_full_finetuning_V2_the_final_baseline.ipynb` in Google Colab
- Run all cells — training takes ~7–8 minutes on a T4 GPU
- The model will be saved automatically to `MyDrive/mt5_finetuned_updated/`

**Option 2 — Download from Google Drive**
- Request access to the shared model folder (contact the repo owner)
- Place the downloaded folder at `MyDrive/mt5_finetuned_updated/` in your Google Drive
- The notebooks will load it from there automatically

### Model files
```
mt5_finetuned_updated/
├── model.safetensors       # Model weights (~300MB) — not in repo
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── tokenizer.model (spiece.model)
└── training_args.bin
```

### Model details

| Property | Value |
|----------|-------|
| Base model | google/mt5-small |
| Task | Binary sentiment classification (Positive / Negative) |
| Dataset | Malayalam–English Code-Mixed (FIRE 2020 – Dravidian CodeMix) |
| Test Accuracy | 84.62% |
| Test Weighted F1 | 81.12% |
| Test Macro F1 | 65.6% |
