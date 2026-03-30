from fastapi import FastAPI, Query
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = FastAPI()

# -------------------------------
# Config
# -------------------------------
MODELS = {
    "distilbert": "distilbert-base-uncased",
    "bert_base": "AventIQ-AI/bert-spam-detection",
    "moe_bert": "AntiSpamInstitute/spam-detector-bert-MoE-v2.2"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset config (can override in TrueFoundry ENV)
DATA_PATH = os.getenv("DATA_PATH", "spam_data.csv")
DATA_URL = os.getenv("DATA_URL", "")  # Google Sheet / S3 link

# Lazy-loaded models cache
loaded_models = {}

# -------------------------------
# Model Loader
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
# Prediction Logic
# -------------------------------
def classify(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    label = torch.argmax(probs).item()
    confidence = probs[0][label].item()

    return {
        "label": "spam" if label == 1 else "not spam",
        "confidence": round(confidence, 4)
    }

# -------------------------------
# Ensemble Logic
# -------------------------------
def ensemble(results):
    votes = [r["label"] for r in results.values()]
    return "spam" if votes.count("spam") > len(votes) / 2 else "not spam"

# -------------------------------
# Dataset Loaders
# -------------------------------
def load_from_csv(path):
    df = pd.read_csv(path)
    return df

def load_from_url(url):
    df = pd.read_csv(url)
    return df

def get_dataset(source="file"):
    if source == "sheet" and DATA_URL:
        return load_from_url(DATA_URL)
    return load_from_csv(DATA_PATH)

# -------------------------------
# Evaluation Logic
# -------------------------------
def evaluate_dataset(df):
    results_summary = {}

    for name, model_path in MODELS.items():
        tokenizer, model = get_model(name, model_path)

        y_true = []
        y_pred = []

        for _, row in df.iterrows():
            text = str(row["text"])
            true_label = int(row["label"])

            pred = classify(text, tokenizer, model)["label"]
            pred_label = 1 if pred == "spam" else 0

            y_true.append(true_label)
            y_pred.append(pred_label)

        results_summary[name] = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred), 4),
            "recall": round(recall_score(y_true, y_pred), 4),
            "f1": round(f1_score(y_true, y_pred), 4),
        }

    return results_summary

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict(data: dict):
    text = data.get("text", "")

    results = {}

    for name, path in MODELS.items():
        tokenizer, model = get_model(name, path)
        results[name] = classify(text, tokenizer, model)

    final_label = ensemble(results)

    return {
        "final_prediction": final_label,
        "models": results
    }

# -------------------------------
# Benchmark Endpoint
# -------------------------------
@app.get("/benchmark")
def benchmark(source: str = Query("file", enum=["file", "sheet"])):
    """
    source:
    - file  → local CSV
    - sheet → Google Sheet / S3 URL
    """
    df = get_dataset(source)

    scores = evaluate_dataset(df)

    # pick best model based on F1
    best_model = max(scores, key=lambda x: scores[x]["f1"])

    return {
        "dataset_size": len(df),
        "metrics": scores,
        "best_model": best_model,
        "note": "F1 score used for best model selection"
    }