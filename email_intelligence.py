from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import pandas as pd
from typing import List
from datetime import datetime
import time
import requests
import zipfile
import io
import tarfile
from fastapi.responses import FileResponse

# -------------------------------
# Load ENV
# -------------------------------
load_dotenv()

app = FastAPI(title="Email Intelligence API")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found")

client = OpenAI(api_key=api_key)

# -------------------------------
# Schemas
# -------------------------------
class EmailAnalysis(BaseModel):
    is_spam: bool
    category: str
    intent: str
    system_email_type: str = None
    priority: str
    entities: List[str]
    confidence: float
    reason: str


class EmailInput(BaseModel):
    subject: str = ""
    sender: str = ""
    body: str


class BulkEmailInput(BaseModel):
    emails: List[EmailInput]


def download_enron_dataset():
    url = "https://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz"

    data_dir = "/tmp/enron"
    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, "enron.tar.gz")

    # Download if not exists
    if not os.path.exists(file_path):
        print("Downloading Enron dataset...")
        r = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(r.content)

    return file_path


def load_enron_emails(sample_size=100):
    file_path = download_enron_dataset()

    extract_path = "/tmp/enron/extracted"

    if not os.path.exists(extract_path):
        print("Extracting dataset...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_path)

    emails = []

    for label in ["spam", "ham"]:
        folder = os.path.join(extract_path, "enron1", label)

        if not os.path.exists(folder):
            continue

        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)

            try:
                with open(file_path, "r", errors="ignore") as f:
                    body = f.read()

                emails.append({
                    "sender": "unknown",
                    "subject": "",
                    "body": body,
                    "label": 1 if label == "spam" else 0
                })

                if len(emails) >= sample_size:
                    return emails

            except:
                continue

    return emails

# -------------------------------
# LLM CALL (SAFE + RETRY)
# -------------------------------
def call_llm(full_text: str, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                timeout=30,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "email_analysis",
                        "schema": EmailAnalysis.model_json_schema()
                    }
                },
                messages=[
                    {
                        "role": "system",
                        "content": """
You are an advanced email intelligence system.

Return strictly valid JSON.

Tasks:
- Detect spam
- Classify category (promotion, phishing, system, personal, update)
- Identify intent (sell, scam, alert, request_action, info)
- Detect system email types (otp, invoice, password_reset, notification)
- Assign priority (low, medium, high)
- Extract entities (links, emails, money, urls)
- Confidence score (0-1)
- Short reasoning
"""
                    },
                    {
                        "role": "user",
                        "content": full_text[:4000]
                    }
                ]
            )

            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            if attempt == retries - 1:
                return {
                    "is_spam": False,
                    "category": "error",
                    "intent": "error",
                    "system_email_type": None,
                    "priority": "low",
                    "entities": [],
                    "confidence": 0.0,
                    "reason": f"LLM error: {str(e)}"
                }
            time.sleep(2)


# -------------------------------
# SINGLE EMAIL ANALYSIS
# -------------------------------
def analyze_email_llm(email: EmailInput):
    if not email.body:
        return {
            "is_spam": False,
            "category": "empty",
            "intent": "none",
            "system_email_type": None,
            "priority": "low",
            "entities": [],
            "confidence": 0.0,
            "reason": "Empty email body"
        }

    full_text = f"""
From: {email.sender}
Subject: {email.subject}
Body: {email.body}
"""

    return call_llm(full_text)


# -------------------------------
# API ROUTES
# -------------------------------
@app.get("/")
def home():
    return {"status": "Email Intelligence API Running"}


@app.post("/analyze-email")
def analyze_email(email: EmailInput):
    return analyze_email_llm(email)


# -------------------------------
# BULK PROCESSING + EXCEL EXPORT
# -------------------------------
@app.post("/analyze-bulk")
def analyze_bulk(data: BulkEmailInput):
    results = []

    for idx, email in enumerate(data.emails):
        result = analyze_email_llm(email)

        results.append({
            "sender": email.sender,
            "subject": email.subject,
            "body": email.body[:200],  # truncate
            **result
        })

    df = pd.DataFrame(results)

    # File name with timestamp
    filename = f"email_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    filepath = f"/tmp/{filename}"

    df.to_excel(filepath, index=False)

    return {
        "message": "Processed successfully",
        "file_path": filepath,
        "total_emails": len(results)
    }


# -------------------------------
# DATASET BENCHMARK (CSV INPUT)
# -------------------------------
@app.post("/benchmark-dataset")
def benchmark_dataset(file_path: str):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    df = pd.read_csv(file_path)

    if "body" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain 'body' column")

    results = []

    for _, row in df.iterrows():
        email = EmailInput(
            subject=row.get("subject", ""),
            sender=row.get("sender", ""),
            body=row.get("body", "")
        )

        result = analyze_email_llm(email)

        results.append({
            "subject": email.subject,
            "sender": email.sender,
            "body": email.body[:200],
            **result
        })

    out_df = pd.DataFrame(results)

    filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    filepath = f"/tmp/{filename}"

    out_df.to_excel(filepath, index=False)

    return {
        "message": "Benchmark completed",
        "file_path": filepath,
        "rows_processed": len(out_df)
    }


@app.get("/benchmark/enron-auto-excel")
def benchmark_enron_auto(sample_size: int = 100):
    emails = load_enron_emails(sample_size)

    results = []

    correct = 0

    for email in emails:
        input_email = EmailInput(
            sender=email["sender"],
            subject=email["subject"],
            body=email["body"]
        )

        result = analyze_email_llm(input_email)

        predicted = 1 if result["is_spam"] else 0
        actual = email["label"]

        is_correct = predicted == actual

        if is_correct:
            correct += 1

        results.append({
            "body": email["body"][:200],
            "actual_label": actual,
            "predicted_label": predicted,
            "correct": is_correct,
            **result
        })

    accuracy = correct / len(results)

    df = pd.DataFrame(results)

    filename = f"enron_llm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    filepath = f"/tmp/{filename}"

    df.to_excel(filepath, index=False)

    return {
        "message": "Enron benchmark completed",
        "file_path": filepath,
        "accuracy": round(accuracy, 4),
        "total_samples": len(results)
    }

@app.get("/download")
def download_file(file_path: str):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )