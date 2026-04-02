import os
import json
import time
import pandas as pd
import requests
import tarfile
from datetime import datetime
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Using the latest Google GenAI SDK for Strict JSON Schema support
from google import genai
from google.genai import types

# 1. INITIALIZATION
load_dotenv(override=True)
app = FastAPI(title="Email Intelligence API - Gemini 3 Edition")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-3-flash-preview"

# 2. INPUT SCHEMAS
class EmailInput(BaseModel):
    subject: str = ""
    sender: str = ""
    body: str

class BulkEmailInput(BaseModel):
    emails: List[EmailInput]

# 3. STRUCTURED OUTPUT SCHEMA (The "Strict Layer")
# This is passed to the model to force perfect JSON formatting
RESPONSE_SCHEMA = {
    "type": "OBJECT",
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
        "intent": {"type": "STRING", "description": "Primary purpose of the email"},
        "is_spam": {"type": "BOOLEAN", "description": "True if promotional, phishing, or unsolicited"},
        "is_system_generated": {"type": "BOOLEAN", "description": "True if automated or no-reply"},
        "reason": {"type": "STRING", "description": "Short explanation for spam/system-gen, else null"},
        "category": {"type": "STRING", "description": "1-3 word lowercase category"},
        "priority": {"type": "STRING", "enum": ["high", "medium", "low"]},
        "confidence": {"type": "NUMBER", "description": "Score 0.0 to 1.0 based on clarity"}
    }
}

# SYSTEM_INSTRUCTION = """
# You are a high-performance email classification engine. 
# Analyze the subject and body to extract intent and metadata.
# Strictly adhere to the provided JSON schema. 
# Be aggressive in identifying spam and marketing materials.
# """

SYSTEM_INSTRUCTION = """
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

# 4. CORE LLM CALL
def call_gemini_structured(subject: str, body: str, retries=2):
    # Clean input: remove null bytes and truncate to prevent token overflow
    clean_body = body.replace("\x00", "").strip()[:6000]
    prompt = f"Subject: {subject}\n\nBody:\n{clean_body}"

    start_time = time.time()
    
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    temperature=0,  # Deterministic output
                )
            )
            
            # The SDK's 'parsed' attribute returns a Python dict automatically
            result = response.parsed
            latency = round(time.time() - start_time, 3)
            
            if isinstance(result, dict):
                result["latency_sec"] = latency
                return result
            else:
                # Fallback if parsing fails
                res_dict = json.loads(response.text)
                res_dict["latency_sec"] = latency
                return res_dict

        except Exception as e:
            if attempt == retries:
                return {
                    "intent": "error", "is_spam": False, "is_system_generated": False,
                    "reason": f"LLM Error: {str(e)}", "category": "error", "priority": "low",
                    "confidence": 0.0, "latency_sec": round(time.time() - start_time, 3)
                }
            time.sleep(1)

# 5. ROUTES
@app.get("/")
def home():
    return {"status": "Online", "engine": MODEL_ID}

@app.post("/analyze-email")
def analyze_email(email: EmailInput):
    if not email.body:
        return {"error": "Email body is required"}
    return call_gemini_structured(email.subject, email.body)

@app.post("/analyze-bulk")
def analyze_bulk(data: BulkEmailInput):
    results = []
    total_latency = 0

    # ThreadPool for faster concurrent processing
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(call_gemini_structured, e.subject, e.body) for e in data.emails]
        for i, future in enumerate(futures):
            res = future.result()
            total_latency += res.get("latency_sec", 0)
            results.append({
                "sender": data.emails[i].sender,
                "subject": data.emails[i].subject,
                **res
            })

    avg_latency = total_latency / len(results) if results else 0
    
    filename = f"bulk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    path = f"/tmp/{filename}"
    os.makedirs("/tmp", exist_ok=True)
    pd.DataFrame(results).to_excel(path, index=False)

    return {
        "download": f"/download/{filename}", 
        "avg_latency_sec": round(avg_latency, 3),
        "total_processed": len(results)
    }

@app.get("/download/{filename}")
def download_file(filename: str):
    path = f"/tmp/{filename}"
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=filename)

# 6. ENRON DATA LOADING HELPERS
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

@app.get("/benchmark/enron-auto")
def benchmark_enron(sample_size: int = 100):
    data = load_enron(sample_size)

    results = []
    correct = 0

    for email in data:
        res = call_gemini_structured(email["subject"], email["body"])

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


# @app.get("/benchmark/enron-auto")
# def benchmark(sample_size: int = 20):
#     tar_path = download_enron()
#     extract_path = "/tmp/enron/data"
#     if not os.path.exists(extract_path):
#         with tarfile.open(tar_path, "r:gz") as tar:
#             tar.extractall(extract_path)

#     data = []
#     for label in ["spam", "ham"]:
#         folder = os.path.join(extract_path, "enron1", label)
#         for file in os.listdir(folder):
#             if len(data) >= sample_size: break
#             with open(os.path.join(folder, file), "r", errors="ignore") as f:
#                 data.append({"body": f.read(), "label": 1 if label == "spam" else 0})

#     results = []
#     correct = 0
#     for item in data:
#         res = call_gemini_structured("", item["body"])
#         pred = 1 if res["is_spam"] else 0
#         if pred == item["label"]: correct += 1
#         results.append({"actual": item["label"], "predicted": pred, **res})

#     return {
#         "accuracy": round(correct / len(data), 4),
#         "processed": len(data),
#         "results": results
#     }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)