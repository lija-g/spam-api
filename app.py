from fastapi import FastAPI
import torch

app = FastAPI()

model = None
tokenizer = None

MODEL_NAME = "AventIQ-AI/bert-spam-detection"

def load_model():
    global model, tokenizer
    if model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def classify(text):
    load_model()  # load only when needed
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)

    label = torch.argmax(probs).item()
    return "spam" if label == 1 else "not spam"

@app.get("/")
def home():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    text = data["text"]
    result = classify(text)
    return {"label": result}
