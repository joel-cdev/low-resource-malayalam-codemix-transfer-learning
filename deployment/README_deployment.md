# Deployment — Malayalam–English Sentiment API

This document explains how to run the fine-tuned mT5-small model as a REST API using FastAPI and Docker.

---

## What This Does

The API loads the trained mT5-small model and exposes these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Predict sentiment for one text |
| `/predict/batch` | POST | Predict sentiment for up to 50 texts |
| `/model/info` | GET | Model details and test results |

---

## Folder Structure

```
deployment/
├── app.py
├── Dockerfile
├── requirements.txt
├── README_deployment.md
├── .gitignore
└── model/                      ← Download from Google Drive (not in GitHub)
    ├── config.json
    ├── generation_config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── training_args.bin
    └── model.safetensors       ← ~1GB model weights
```

---

## Step 1 — Get the Model

The trained model is stored on Google Drive (too large for GitHub).

Download `mt5_finetuned_updated` from Google Drive, rename the folder to `model`, and place it inside the `deployment/` folder.

---

## Step 2 — Build the Docker Image

Open a terminal inside the `deployment/` folder and run:

```bash
docker build -t malayalam-sentiment-api .
```

This takes 3–5 minutes on first build.

---

## Step 3 — Run the Container

```bash
docker run -p 8000:8000 malayalam-sentiment-api
```

You should see:
```
Loading model from ./model ...
Model loaded successfully.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Step 4 — Test the API

### Browser (Swagger UI — recommended for demo)
```
http://localhost:8000/docs
```

### Single prediction
```json
POST /predict
{
  "text": "Amazing movie, super acting!"
}
```
Response:
```json
{
  "text": "Amazing movie, super acting!",
  "sentiment": "positive",
  "inference_time_ms": 95.18
}
```

### Batch prediction
```json
POST /predict/batch
{
  "texts": [
    "Amazing movie, super acting!",
    "Worst movie ever. Waste of time.",
    "Nalla padam, must watch!",
    "Very bad film, not recommended"
  ]
}
```

---

## Sample Inputs and Expected Outputs

| Input Text | Language | Prediction |
|-----------|----------|------------|
| Amazing movie, super acting! | English | positive |
| Worst movie ever. Waste of time. | English | negative |
| Nalla padam, must watch! | Malayalam–English | positive |
| Adipoli film, superb performance! | Malayalam–English | positive |
| Mosham cinema, samayam oru waste | Malayalam–English | negative |

---

## How Inference Works

The API uses **logit-based classification** rather than free-form text generation. A single forward pass is performed and the raw logit scores for the `"positive"` token (id: 18205) and `"negative"` token (id: 259) are compared directly. This replicates the internal evaluation method used by HuggingFace's `Seq2SeqTrainer`.

---

## Stop the Container

Press `Ctrl+C` in the terminal, or:

```bash
docker ps
docker stop <container_id>
```

---

## Requirements

- Docker Desktop installed and running
- At least 3GB free disk space
- The `model/` folder downloaded from Google Drive
