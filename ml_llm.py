from fastapi import FastAPI, Query
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

# -------------------------------
# CONFIG
# -------------------------------
MODELS = {
    "bert_base": "AventIQ-AI/bert-spam-detection"
}

device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = os.getenv("DATA_PATH", "spam_data.csv")

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

loaded_models = {}

# -------------------------------
# Pydantic Schema (LLM Output)
# -------------------------------
class SpamAnalysis(BaseModel):
    is_spam: bool
    category: str
    intent: str
    confidence: float
    reason: str

# -------------------------------
# MODEL LOADER
# -------------------------------
def get_model(name, path):
    if name not in loaded_models:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)

        model.to(device)
        model.eval()

        loaded_models[name] = (tokenizer, model)

    return loaded_models[name]

# -------------------------------
# ML Prediction
# -------------------------------
def classify_ml(text, tokenizer, model):
    if not text:
        return "not spam"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    label = torch.argmax(probs).item()

    return "spam" if label == 1 else "not spam"

# -------------------------------
# LLM Prediction (Structured)
# -------------------------------
def classify_llm(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "spam_analysis",
                    "schema": SpamAnalysis.model_json_schema()
                }
            },
            messages=[
                {
                    "role": "system",
                    "content": "You are a spam detection system. Classify email into spam/not spam, category, intent, and confidence."
                },
                {
                    "role": "user",
                    "content": text[:3000]  # prevent token overflow
                }
            ]
        )

        return response.choices[0].message.parsed

    except Exception as e:
        return {
            "is_spam": False,
            "category": "unknown",
            "intent": "unknown",
            "confidence": 0.0,
            "reason": str(e)
        }

# -------------------------------
# LOAD DATASET
# -------------------------------
def load_data():
    return pd.read_csv(DATA_PATH)

# -------------------------------
# EVALUATE ML
# -------------------------------
def evaluate_ml(df):
    tokenizer, model = get_model("bert_base", MODELS["bert_base"])

    y_true, y_pred = [], []

    for _, row in df.iterrows():
        text = str(row["text"])
        true_label = int(row["label"])

        pred = classify_ml(text, tokenizer, model)
        pred_label = 1 if pred == "spam" else 0

        y_true.append(true_label)
        y_pred.append(pred_label)

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
    }

# -------------------------------
# FULL COMPARISON (ROW LEVEL)
# -------------------------------
def compare_models(df, limit=None):
    tokenizer, model = get_model("bert_base", MODELS["bert_base"])

    results = []

    if limit:
        df = df.head(limit)

    for _, row in df.iterrows():
        text = str(row["text"])
        true_label = "spam" if int(row["label"]) == 1 else "not spam"

        # ML
        ml_pred = classify_ml(text, tokenizer, model)

        # LLM
        llm_result = classify_llm(text)

        results.append({
            "text": text,
            "true_label": true_label,
            "ml_prediction": ml_pred,
            "ml_correct": ml_pred == true_label,
            "llm_prediction": "spam" if llm_result["is_spam"] else "not spam",
            "llm_correct": ("spam" if llm_result["is_spam"] else "not spam") == true_label,
            "llm_category": llm_result["category"],
            "llm_intent": llm_result["intent"],
            "llm_confidence": llm_result["confidence"],
            "llm_reason": llm_result["reason"]
        })

    return results

# -------------------------------
# ROUTES
# -------------------------------
@app.get("/")
def home():
    return {"status": "ML + LLM benchmark running"}

# -------------------------------
# ML METRICS ONLY
# -------------------------------
@app.get("/benchmark/ml")
def benchmark_ml():
    df = load_data()
    metrics = evaluate_ml(df)

    return {
        "model": "bert_base",
        "metrics": metrics
    }

# -------------------------------
# FULL COMPARISON
# -------------------------------
@app.get("/benchmark/compare")
def benchmark_compare(limit: int = Query(20)):
    df = load_data()

    results = compare_models(df, limit=limit)

    return {
        "samples": len(results),
        "results": results
    }