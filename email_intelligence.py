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
import tarfile
from fastapi.responses import FileResponse

# -------------------------------
# LOAD ENV
# -------------------------------
load_dotenv(override=True)

app = FastAPI(title="Email Intelligence API")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found")

client = OpenAI(api_key=api_key)

# -------------------------------
# INPUT SCHEMAS
# -------------------------------
class EmailInput(BaseModel):
    subject: str = ""
    sender: str = ""
    body: str


class BulkEmailInput(BaseModel):
    emails: List[EmailInput]


# -------------------------------
# RESPONSE SCHEMA (STRICT)
# -------------------------------
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "intent",
        "is_spam",
        "is_system_generated",
        "reason",
        "category",
        "priority",
        "confidence"
    ],
    "properties": {
        "intent": {
            "type": "string",
            "description": "Primary purpose of the email",
            "maxLength": 80
        },
        "is_spam": {
            "type": "boolean",
            "description": "True if promotional or phishing"
        },
        "is_system_generated": {
            "type": "boolean",
            "description": "True if automated email"
        },
        "reason": {
            "type": ["string", "null"],
            "description": "Reason when spam/system-generated",
            "maxLength": 120
        },
        "category": {
            "type": "string",
            "description": "1-3 word category",
            "pattern": "^[a-z ]{1,30}$"
        },
        "priority": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "Urgency level"
        },
        "confidence": {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "description": "Confidence score between 0 and 1 based on clarity and certainty"
        }
        
    }
}

# -------------------------------
# SYSTEM PROMPT (FULL)
# -------------------------------
SYSTEM_PROMPT = """
You are an email classification engine.

Task:
Given an email subject and body, classify the email and return only valid JSON that matches the required schema.

Rules:
- Use only the subject and body provided.
- Do not infer missing facts.
- If multiple intents exist, choose the most dominant one.
- Keep all text fields short and consistent.
- Output must be valid JSON only. No markdown, no explanation.

Classification rules:

intent:
- Return a short phrase describing the main purpose of the email.

is_spam:
- true if promotional, unsolicited, phishing, or irrelevant bulk mail.
- otherwise false.

is_system_generated:
- true if automated or no-reply emails.
- otherwise false.

reason:
- If is_spam or is_system_generated is true, return a short explanation.
- Otherwise return null.

category:
- 1–3 words, lowercase.

priority:
- high → urgent/blocking
- medium → issue exists
- low → informational/spam/system-generated
confidence:
- Return a number between 0 and 1
- High (0.8–1.0): clear intent, strong signals
- Medium (0.5–0.8): somewhat clear but ambiguous
- Low (0.0–0.5): unclear, noisy, or insufficient data

Return exactly this JSON shape:
{
  "intent": string,
  "is_spam": boolean,
  "is_system_generated": boolean,
  "reason": string | null,
  "category": string,
  "priority": "high" | "medium" | "low"
}
"""

# -------------------------------
# LLM CALL
# -------------------------------
def call_llm(subject: str, body: str, retries=3):
    user_prompt = f"""
Email Subject: {subject}
Email Body: {body[:3000]}
"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                temperature=0,
                timeout=30,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "email_classifier",
                        "schema": SCHEMA
                    }
                },
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            if attempt == retries - 1:
                return {
                    "intent": "error",
                    "is_spam": False,
                    "is_system_generated": False,
                    "reason": str(e),
                    "category": "error",
                    "priority": "low",
                    "confidence": 0.0
                }
            time.sleep(1)


# -------------------------------
# EMAIL ANALYSIS
# -------------------------------
def analyze_email_llm(email: EmailInput):
    if not email.body:
        return {
            "intent": "empty",
            "is_spam": False,
            "is_system_generated": False,
            "reason": "empty body",
            "category": "none",
            "priority": "low",
            "confidence": 0.0
        }

    return call_llm(email.subject, email.body)


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
# BULK + EXCEL EXPORT
# -------------------------------
@app.post("/analyze-bulk")
def analyze_bulk(data: BulkEmailInput):
    results = []

    for email in data.emails:
        result = analyze_email_llm(email)

        results.append({
            "sender": email.sender,
            "subject": email.subject,
            "body": email.body[:200],
            **result
        })

    df = pd.DataFrame(results)

    filename = f"email_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    filepath = f"/tmp/{filename}"

    df.to_excel(filepath, index=False)

    return {
        "message": "Processed successfully",
        "download": f"/download/{filename}",
        "total_emails": len(results)
    }


# -------------------------------
# DOWNLOAD
# -------------------------------
@app.get("/download/{filename}")
def download_file(filename: str):
    path = f"/tmp/{filename}"

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path, filename=filename)


# -------------------------------
# ENRON AUTO DOWNLOAD
# -------------------------------
def download_enron():
    url = "https://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz"

    os.makedirs("/tmp/enron", exist_ok=True)
    file_path = "/tmp/enron/enron.tar.gz"

    if not os.path.exists(file_path):
        r = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(r.content)

    return file_path


def load_enron(sample_size=100):
    tar_path = download_enron()
    extract_path = "/tmp/enron/data"

    if not os.path.exists(extract_path):
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_path)

    emails = []

    for label in ["spam", "ham"]:
        folder = os.path.join(extract_path, "enron1", label)

        if not os.path.exists(folder):
            continue

        for file in os.listdir(folder):
            try:
                with open(os.path.join(folder, file), "r", errors="ignore") as f:
                    body = f.read()

                emails.append({
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
# ENRON BENCHMARK + EXCEL
# -------------------------------
@app.get("/benchmark/enron-auto")
def benchmark_enron(sample_size: int = 100):
    data = load_enron(sample_size)

    results = []
    correct = 0

    for email in data:
        res = call_llm(email["subject"], email["body"])

        pred = 1 if res["is_spam"] else 0
        actual = email["label"]

        if pred == actual:
            correct += 1

        results.append({
            "body": email["body"][:200],
            "actual": actual,
            "predicted": pred,
            "correct": pred == actual,
            **res
        })

    accuracy = correct / len(results)

    df = pd.DataFrame(results)

    filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    path = f"/tmp/{filename}"

    df.to_excel(path, index=False)

    return {
        "message": "Benchmark completed",
        "accuracy": round(accuracy, 4),
        "download": f"/download/{filename}",
        "total_samples": len(results)
    }