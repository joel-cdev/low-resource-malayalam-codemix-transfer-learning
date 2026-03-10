import time
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Malayalam-English Sentiment Analysis API",
    description="Binary sentiment analysis (positive/negative) for Malayalam-English code-mixed text using fine-tuned mT5-small.",
    version="1.0.0"
)

MODEL_DIR = os.getenv("MODEL_DIR", "./model")

# ── Load model at startup ─────────────────────────────────────────────────────
print(f"Loading model from {MODEL_DIR} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
model.eval()

# Pre-compute label token IDs for logit-based classification
# This avoids generation entirely and directly compares label scores
POS_TOKEN_ID = tokenizer.encode("positive", add_special_tokens=False)[0]
NEG_TOKEN_ID = tokenizer.encode("negative", add_special_tokens=False)[0]
print(f"Label token IDs — positive: {POS_TOKEN_ID}, negative: {NEG_TOKEN_ID}")

# Decoder start token (required for mT5 seq2seq forward pass)
DECODER_START_ID = tokenizer.pad_token_id
print("Model loaded successfully.")


# ── Schemas ───────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

class PredictionOutput(BaseModel):
    text: str
    sentiment: str
    inference_time_ms: float

class BatchOutput(BaseModel):
    results: List[PredictionOutput]


# ── Inference helper ──────────────────────────────────────────────────────────
def predict_sentiment(text: str) -> tuple:
    """
    Logit-based classification: instead of generating tokens freely,
    we do a single forward pass and compare the logit scores for
    'positive' vs 'negative' at the first decoder position.
    This is more reliable than generation for classification tasks.
    """
    start = time.time()

    formatted = f"sentiment: {text}"
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    decoder_input_ids = torch.tensor([[DECODER_START_ID]])

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids
        )

    first_token_logits = outputs.logits[0, 0, :]
    pos_score = first_token_logits[POS_TOKEN_ID].item()
    neg_score = first_token_logits[NEG_TOKEN_ID].item()

    sentiment = "negative" if pos_score >= neg_score else "positive"
    elapsed_ms = (time.time() - start) * 1000

    return sentiment, round(elapsed_ms, 2)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model": "mT5-small fine-tuned (Malayalam-English sentiment)",
        "label_tokens": {"positive": POS_TOKEN_ID, "negative": NEG_TOKEN_ID}
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    sentiment, latency = predict_sentiment(input.text)
    return PredictionOutput(
        text=input.text,
        sentiment=sentiment,
        inference_time_ms=latency
    )


@app.post("/predict/batch", response_model=BatchOutput)
def predict_batch(input: BatchInput):
    if not input.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty.")
    if len(input.texts) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 texts per batch.")
    results = []
    for text in input.texts:
        sentiment, latency = predict_sentiment(text)
        results.append(PredictionOutput(text=text, sentiment=sentiment, inference_time_ms=latency))
    return BatchOutput(results=results)


@app.get("/model/info")
def model_info():
    return {
        "base_model": "google/mt5-small",
        "task": "Binary sentiment classification (positive / negative)",
        "language": "Malayalam-English code-mixed",
        "dataset": "FIRE 2020 Dravidian CodeMix",
        "train_size": 4036,
        "test_results": {
            "accuracy": 0.8462,
            "weighted_f1": 0.8112,
            "macro_f1": 0.6559
        },
        "inference_method": "logit-based classification (pos vs neg token scores)"
    }
