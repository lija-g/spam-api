from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# -------------------------------
# Config
# -------------------------------
MODELS = {
    "distilbert": "mrm8488/distilbert-base-uncased-finetuned-sms-spam-detection",
    "bert_base": "AventIQ-AI/bert-spam-detection",
     "moe_bert": "AntiSpamInstitute/spam-detector-bert-MoE-v2.2"
    # ⚠️ Avoid MoE in production unless you have high GPU memory
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy-loaded models
loaded_models = {}

# -------------------------------
# Model Loader (Lazy Loading)
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
# Ensemble Logic (Majority Vote)
# -------------------------------
def ensemble(results):
    votes = [r["label"] for r in results.values()]

    if votes.count("spam") > len(votes) / 2:
        return "spam"
    return "not spam"

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
# Benchmark Dataset (Small Demo)
# -------------------------------
test_data = [
    {"text": "Win a free iPhone now!!!", "label": 1},
    {"text": "Congratulations! You won a lottery", "label": 1},
    {"text": "Let's have a meeting tomorrow", "label": 0},
    {"text": "Please review the attached document", "label": 0},
    {"text": "Claim your free prize now", "label": 1},
    {"text": "Project deadline is next week", "label": 0},
]

# -------------------------------
# Benchmark Logic
# -------------------------------
def evaluate():
    scores = {}

    for name, path in MODELS.items():
        tokenizer, model = get_model(name, path)

        correct = 0

        for item in test_data:
            pred = classify(item["text"], tokenizer, model)["label"]
            true_label = "spam" if item["label"] == 1 else "not spam"

            if pred == true_label:
                correct += 1

        accuracy = correct / len(test_data)
        scores[name] = round(accuracy, 4)

    return scores

# -------------------------------
# Benchmark Endpoint (SAFE)
# -------------------------------
@app.get("/benchmark")
def benchmark():
    scores = evaluate()
    best_model = max(scores, key=scores.get)

    return {
        "accuracy": scores,
        "best_model": best_model,
        "note": "Use larger dataset for real accuracy"
    }
