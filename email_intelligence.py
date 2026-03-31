from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json


load_dotenv()
app = FastAPI()

api_key = os.getenv("OPENAI_API_KEY")
import os
print("ENV FILE EXISTS:", os.path.exists(".env"))

print("API Key:", api_key)

# -------------------------------
# OpenAI Client
# -------------------------------
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)

# -------------------------------
# Pydantic Schema
# -------------------------------
class EmailAnalysis(BaseModel):
    is_spam: bool
    category: str  # promotion, phishing, system, personal, update, etc.
    intent: str  # sell, alert, request_action, scam, information
    system_email_type: str = None  # otp, invoice, alert, notification
    priority: str  # low, medium, high
    entities: list[str]
    confidence: float
    reason: str

# -------------------------------
# Input Schema
# -------------------------------
class EmailInput(BaseModel):
    subject: str = ""
    sender: str = ""
    body: str

# -------------------------------
# LLM Analyzer
# -------------------------------
def analyze_email_llm(email: EmailInput):
    full_text = f"""
    From: {email.sender}
    Subject: {email.subject}
    Body: {email.body}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
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

Classify the email into:
- spam or not
- category (promotion, phishing, system, personal, update)
- intent (sell, scam, alert, request_action, info)
- detect system email types (otp, invoice, password_reset, notification)
- assign priority (low, medium, high)
- extract key entities (links, money, emails, urls)
- give confidence score (0 to 1)
- explain reasoning briefly

Be accurate and structured.
"""
            },
            {
                "role": "user",
                "content": full_text[:4000]  # prevent overflow
            }
        ]
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    return data
# -------------------------------
# API Routes
# -------------------------------
@app.get("/")
def home():
    return {"status": "LLM Email Intelligence Running"}

@app.post("/analyze-email")
def analyze_email(email: EmailInput):
    result = analyze_email_llm(email)
    return result