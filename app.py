from fastapi import FastAPI, Query
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tarfile
import requests
import random
from email import policy
from email.parser import BytesParser

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

DATA_PATH = os.getenv("DATA_PATH", "spam_data.csv")
DATA_URL = os.getenv("DATA_URL", "")

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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512 )
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
    return pd.read_csv(path)

def load_from_url(url):
    return pd.read_csv(url)

def get_dataset(source="file"):
    if source == "sheet" and DATA_URL:
        return load_from_url(DATA_URL)
    return load_from_csv(DATA_PATH)

# -------------------------------
# Evaluation (Summary Metrics)
# -------------------------------
def evaluate_dataset(df):
    results_summary = {}

    for name, model_path in MODELS.items():
        tokenizer, model = get_model(name, model_path)

        y_true, y_pred = [], []

        for _, row in df.iterrows():
            text = build_full_text(row)
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
# Evaluation (Detailed Per Row)
# -------------------------------
def evaluate_detailed(df):
    detailed_results = []

    for _, row in df.iterrows():
        text = build_full_text(row)
        true_label = "spam" if int(row["label"]) == 1 else "not spam"

        model_outputs = {}

        for name, path in MODELS.items():
            tokenizer, model = get_model(name, path)
            pred = classify(text, tokenizer, model)

            model_outputs[name] = {
                **pred,
                "correct": pred["label"] == true_label
            }

        # Ensemble prediction
        ensemble_pred = ensemble(model_outputs)

        detailed_results.append({
            "text": text,
            "true_label": true_label,
            "ensemble_prediction": ensemble_pred,
            "ensemble_correct": ensemble_pred == true_label,
            "models": model_outputs
        })

    return detailed_results

ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
ENRON_DIR = "./enron_data"

def download_enron():
    os.makedirs(ENRON_DIR, exist_ok=True)

    tar_path = os.path.join(ENRON_DIR, "enron.tar.gz")

    # Download
    if not os.path.exists(tar_path):
        print("Downloading Enron dataset...")
        r = requests.get(ENRON_URL, stream=True)
        with open(tar_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)

    # Extract
    extract_path = os.path.join(ENRON_DIR, "maildir")

    if not os.path.exists(extract_path):
        print("Extracting Enron dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=ENRON_DIR)

    return extract_path

def parse_email(file_path):
    try:
        with open(file_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        body = ""
        if msg.get_body():
            body = msg.get_body(preferencelist=('plain', 'html')).get_content()

        return {
            "from": str(msg["from"] or ""),
            "subject": str(msg["subject"] or ""),
            "text": body
        }
    except:
        return None

def build_enron_dataset(sample_size=200):
    base_path = download_enron()

    file_paths = []

    for root, _, files in os.walk(base_path):
        for file in files:
            file_paths.append(os.path.join(root, file))

    sampled_files = random.sample(file_paths, min(sample_size, len(file_paths)))

    data = []

    for path in sampled_files:
        email_data = parse_email(path)

        if email_data:
            text_lower = email_data["text"].lower()

            # ⚠️ heuristic labeling (not perfect)
            label = 1 if any(word in text_lower for word in ["free", "win", "money", "offer"]) else 0

            data.append({
                "from": email_data["from"],
                "subject": email_data["subject"],
                "text": email_data["text"],
                "label": label
            })

    return pd.DataFrame(data)

def build_full_text(row):
    subject = str(row.get("subject", ""))[:200]
    sender = str(row.get("from", ""))[:100]
    body = str(row.get("text", ""))[:1000]  # limit body

    return f"""
    From: {sender}
    Subject: {subject}
    Body: {body}
    """
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
# Combined Benchmark Endpoint
# -------------------------------
@app.get("/benchmark")
def benchmark(
    source: str = Query("file", enum=["file", "sheet"]),
    detailed: bool = Query(False),
    limit: int = Query(None)
):
    """
    source:
      - file  → local CSV
      - sheet → Google Sheet / S3 URL

    detailed:
      - true → include per-row predictions

    limit:
      - limit number of rows (for speed)
    """

    df = get_dataset(source)

    if limit:
        df = df.head(limit)

    # Summary metrics
    scores = evaluate_dataset(df)
    best_model = max(scores, key=lambda x: scores[x]["f1"])

    response = {
        "dataset_size": len(df),
        "metrics": scores,
        "best_model": best_model
    }

    # Add detailed results if requested
    if detailed:
        response["detailed_results"] = evaluate_detailed(df)

    return response


@app.get("/benchmark/enron-auto")
def benchmark_enron_auto(
    sample_size: int = Query(200),
    detailed: bool = Query(True)
):
    """
    Auto-download + benchmark Enron dataset
    """

    df = build_enron_dataset(sample_size=sample_size)

    scores = evaluate_dataset(df)
    best_model = max(scores, key=lambda x: scores[x]["f1"])

    response = {
        "dataset": "Enron Auto (heuristic labels)",
        "dataset_size": len(df),
        "metrics": scores,
        "best_model": best_model,
        "note": "Labels are heuristic (approximate accuracy)"
    }

    if detailed:
        response["detailed_results"] = evaluate_detailed(df)

    return response